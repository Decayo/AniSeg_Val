#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : mclane.py

from datasets.BaseDataset import BaseDataset
from config import config

class AniSeg(BaseDataset):
    @classmethod
    def get_class_names(*args):
        if(config.num_classes == 13):
            return ['00_unlabeled',
                    '01_hair',
                    '02_face', 
                    '03_eyes',
                    "04_assesories",
                    '05_ears',
                    '06_torso', 
                    '07_torso_wearing', 
                    '08_arms', 
                    '09_hands',
                    '10_legs',
                    '11_feet', 
                '12_legs_wearing/decoration']
        return ['00_unlabeled','01_hair', '02_hair_decoration', '03_face', '04_eyes','05_mouth',
                '06_face_wearing/decoration',
                '07_ears', '08_torso', '09_torso_wearing', '10_arms', '11_hands',
                '12_legs',
                '13_feet', '14_legs_wearing/decoration', '15_stockings',
                '16_shoes',
                ]
    @classmethod
    def get_class_colors(*args):
        # def uint82bin(n, count=8):
        #     """returns the binary of integer n, count refers to amount of bits"""
        #     return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        # N = 17
        # cmap = np.zeros((N, 3), dtype=np.uint8)
        # for i in range(N):
        #     r, g, b = 0, 0, 0
        #     id = i
        #     for j in range(7):
        #         str_id = uint82bin(id)
        #         r = r ^ (np.uint8(str_id[-1]) << (7 - j))
        #         g = g ^ (np.uint8(str_id[-2]) << (7 - j))
        #         b = b ^ (np.uint8(str_id[-3]) << (7 - j))
        #         id = id >> 3
        #     cmap[i, 0] = r
        #     cmap[i, 1] = g
        #     cmap[i, 2] = b
        # class_colors = cmap.tolist()
        # return class_colors
        if(config.num_classes == 13):
            return [[0, 0, 0],
                    [255, 0, 0], 
                    [255, 255, 0], 
                    [255, 97, 0], 
                    [245, 222, 179],
                    [218, 165, 105], 
                    [127, 255, 0], 
                    [0, 255, 0],
                    [8, 46, 84],
                    [64, 224, 208], 
                    [176, 226, 255], 
                    [0, 139, 139],
                    [144, 238, 144]]
        return [[0, 0, 0],[255, 0, 0], [255, 0, 255], [255, 255, 0], [225, 97, 0],
                [255, 153, 18], [245, 222, 179], [218, 165, 105],
                [127, 255, 0],
                [0, 255, 0], [8, 46, 84], [64, 224, 208],
                [176, 226, 255],
                [0, 139, 139], [144, 238, 144], [139, 101, 8],
                [74, 112, 139]]


