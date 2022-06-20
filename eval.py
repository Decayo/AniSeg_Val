#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from dataloader import AniSeg
from dataloader import ValPre
from network import Network

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:

    azure = False
import time

logger = get_logger()

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from PIL import Image

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]

def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    N = config.num_classes
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    print(class_colors)
    return class_colors[1:]


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}



        fn = name + '.png'

        'save colored result'
        result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        class_colors = AniSeg.get_class_colors()
        palette_list = list(np.array(class_colors).flat)
        if len(palette_list) < 768:
            palette_list += [0] * (768 - len(palette_list))
        result_img.putpalette(palette_list)
        #result_img.save(os.path.join(self.save_path+'_color', fn))

        'save raw result'
        
        # if(self.Epoch is not None):
        #     cv2.imwrite(os.path.join(self.save_path+'_'+self.Epoch, fn), pred)
        # else:
        #     cv2.imwrite(os.path.join(self.save_path, fn), pred)
        #logger.info('Save the image ' + fn)

        if self.show_image:
            colors = AniSeg.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
            fn = name + '.png'
            
            sp = os.path.join(config.save_val_path + r'\log\im_compare_'+os.path.basename(config.eval_source)+'_'+str(self.Epoch))
            os.makedirs(sp, exist_ok=True)
            cv2.imwrite(os.path.join(sp, fn), comp_img)
            #cv2.imshow('comp_image', comp_img)
            #cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        print(len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--val_src','-val',default="easy_val.txt",type = str)
    parser.add_argument('--aspp',default = False,action='store_true')
    parser.add_argument('--cls',default=16,type = int)
    
    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    config.num_classes = args.cls + 1
    config.volna = r'J:\master_1_down\_thesis\code\main_workspace\AniSeg_Val\final_'+str(args.cls)+'_val'
    config.dataset_path = config.volna 
    config.eval_source = os.path.join(config.dataset_path, args.val_src)
    
    config.use_aspp = args.aspp
    config.save_val_path = args.epochs
    
    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    if not os.path.exists( config.save_val_path +'log/'):
        os.makedirs( config.save_val_path +'log/')
    config.val_log_file = config.save_val_path +'log/' + '/val_' + os.path.basename(config.eval_source).replace('.', '-')  +'_' + exp_time +  '.log'
    config.link_val_log_file =  config.val_log_file
    network = Network(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.dataset_path,
                    'gt_root': config.dataset_path,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    print(data_setting)
    val_pre = ValPre()
    dataset = AniSeg(data_setting, 'val', val_pre, training=False)
    
    
    # get all epoch in list
    e_list = []
    for _,_,files in os.walk(args.epochs):
        for ep in files:
            if "epoch" not in ep:
                continue
            e_list.append(str(os.path.basename(str
                                               (ep))))
    
    for e_num in e_list:
        with torch.no_grad():
            segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                    config.image_std, network,
                                    config.eval_scale_array, config.eval_flip,
                                    all_dev, args.verbose,
                                    args.show_image,Epoch=e_num)   
            segmentor.run(config.snapshot_dir, args.epochs+e_num, config.val_log_file,
                        config.link_val_log_file)
#python eval.py -d 0 -e .//epochs//0528_nc_re_a100// -val easy_val.txt --cls 12 -s