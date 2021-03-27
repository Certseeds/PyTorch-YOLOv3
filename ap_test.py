#!/usr/bin/env python3
# coding=utf-8
import argparse
from typing import Tuple, List

import matplotlib.pyplot as plt
from utils.ap import get_dataset_ap
from numpy import trapz

from utils.datasets import ListDataset
from utils.transforms import DEFAULT_TRANSFORMS


def init_parser():
    parser = argparse.ArgumentParser(description='cal mp test')
    parser.add_argument("--img_list_file_path", default="./data/barcode/CorrectDetect.txt", type=str,
                        help="a file path")
    parser.add_argument("--dataset", default="barcode", choices=['VOC', 'COCO', 'barcode'], type=str,
                        help="You know the rules")
    parser.add_argument("--pred_label_path", type=str, help="You know the rules")
    args_r = parser.parse_args()
    return parser, args_r


parser, args = init_parser()


def main() -> None:
    testset = ListDataset(args.img_list_file_path,
                          img_size=416,
                          multiscale=False,
                          transform=DEFAULT_TRANSFORMS)
    # get_dataset_ap(testset, "test/barcode5/labels")
    pres, recas = get_dataset_ap(testset, args.pred_label_path)
    draw(pres, recas)


def draw(pres: List[float], recas: List[float]) -> float:
    recas.insert(0, 0)
    pres.insert(0, 1)
    print(recas, pres)
    plt.plot(recas, pres)
    plt.axis([0, 1.1, 0, 1.1])
    plt.xlabel('recas')
    plt.ylabel('ppres')
    plt.show()
    x = trapz(pres, recas, dx=0.0001)
    print(x)
    return 1


if __name__ == '__main__':
    main()
