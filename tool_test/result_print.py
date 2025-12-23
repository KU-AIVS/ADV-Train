
import os
import logging
import argparse
from pathlib import Path
import cv2
from util import dataset, transform, config
import re
import ast
from collections import defaultdict
cv2.ocl.setUseOpenCL(False)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--attack',default=None)
    parser.add_argument('--at_iter', type=int, default=None)
    parser.add_argument('--avg', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=None)
    parser.add_argument('--train_num', type=int, default=None)
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None


    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.attack = args.attack
    cfg.at_iter = args.at_iter
    cfg.avg =args.avg
    cfg.train_num = args.train_num
    cfg.num_epoch = args.num_epoch

    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger(log_path=None):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_path is not None:
        file_handler = logging.FileHandler(f'{log_path}/log.log', mode='a')  # 파일명 지정
    else:
        file_handler = logging.FileHandler(f'{args.save_folder}/log.log', mode='a')# 파일명 지정
    file_handler.setFormatter(formatter)  # 동일한 포맷 적용
    logger.addHandler(file_handler)  # 로거에 핸들러 추가

    return logger


def result_one_train(save_folder):
    attacks = os.listdir(save_folder)
    attack_results ={}
    for attack in attacks:
        if os.path.isdir(os.path.join(save_folder, attack)):
            log_path = os.path.join(os.path.join(save_folder, attack), 'log.log')
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'Eval result' in line:
                        last_eval_line = line.strip()
            miou = last_eval_line.split('/')[-3].split(' ')[-1]
            attack_results[attack] = float(miou)

    return attack_results


def result_avg_test(BASE_DIR):
    total_metrics = defaultdict(float)
    avg_num = 0
    dict_pattern = re.compile(r"(\{'.*\})")
    for idx in range(1, len(os.listdir(BASE_DIR))):
        if os.path.isdir(str(args.save_folder).replace("None", str(idx))):
            log_path = os.path.join(str(args.save_folder).replace("None", str(idx)), 'log.log')
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = dict_pattern.search(line)
                    if match:
                        dict_string = match.group(1).strip()
                        result_dict = ast.literal_eval(dict_string)
            logger.info("Train {} results: {}".format(idx, result_dict))

            for key, value in result_dict.items():
                if isinstance(value, (int, float)):
                    total_metrics[key] += value
            avg_num += 1

    average_metrics = {}
    for key, total_value in total_metrics.items():
        average_metrics[key] = total_value / avg_num

    return average_metrics


def main():
    global args, logger
    args = get_parser()
    if args.attack is not None:
        args.save_folder = Path(
            args.save_folder.format(attack=args.attack, at_iter=args.at_iter, train_num=args.train_num,
                                    num_epoch=args.num_epoch))
    else:
        args.save_folder = Path(
            args.save_folder.format(train_num=args.train_num, num_epoch=args.num_epoch))
    if args.train_num is not None:
        logger = get_logger()
        logger.info("Train {} loading results....".format(args.train_num))
        attack_results= result_one_train(args.save_folder)
        logger.info("=> Train {} Results".format(args.train_num))
        logger.info(attack_results)

    else:
        BASE_DIR = args.save_folder.parent.parent.parent.parent
        logger = get_logger(BASE_DIR)
        average_metrics= result_avg_test(BASE_DIR)
        formatted_parts = [
            f"'{key}': {average_metrics[key]:.4f}"
            for key in sorted(average_metrics.keys())
        ]
        metrics_line = ", ".join(formatted_parts)
        logger.info(f"Average {metrics_line}")




if __name__ == '__main__':
    main()
