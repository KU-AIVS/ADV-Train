
import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from model.pspnet import PSPNet, DeepLabV3
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize
from attack.attacker import attacker

cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None


    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg



def find_result(file_path):
    log_path = os.path.join(file_path, 'log.log')
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Eval result' in line:
                last_eval_line = line.strip()
    miou = last_eval_line.split('/')[-3].split(' ')[-1]
    return miou

def main():
    global args, logger
    args = get_parser()

    # attacks = ['cw20', 'df20', 'bim20', 'pgd10', 'pgd20', 'pgd40','pgd100']
    attacks = ['cw20', 'bim20', 'pgd10', 'pgd20', 'pgd40','pgd100']

    attack_result ={}

    attack_result['clean'] = find_result(args.save_folder)
    for attack in attacks:
        result_path = os.path.join(args.save_folder, attack)
        attack_result[attack] = find_result(result_path)
    print(attack_result)



if __name__ == '__main__':
    main()
