'''
Author: your name
Date: 2021-03-16 23:48:38
LastEditTime: 2021-03-17 00:02:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \PyTorch-YOLOv3\data\barcode_All\random_train_valid_devide.py
'''
import os
from random import shuffle
import re


def main():
    print(os.path.abspath("./"))
    this_before = "./../../"
    root_before = "../"
    repo_paths = ["barcode_detection_dataset/pictures/images/20210316_2/",
                  "barcode_detection_dataset/pictures/images/20210317_2/",
                  "barcode_detection_dataset/1d_barcode_extended/images/JPEGImages/",
                  "barcode_detection_dataset/1d_barcode_extended_plain/images/Original/",
                  "barcode_detection_dataset/BarcodeDatasets/images/Dataset1/",
                  "barcode_detection_dataset/BarcodeDatasets/images/Dataset2/"]
    img_list_sum = []
    for repo_path in repo_paths:
        img_list = []
        path = this_before + root_before + repo_path
        for entry in os.scandir(path):
            if not os.path.isdir(entry.path):
                img_list.append(entry.name)
        img_list = [root_before + repo_path + x for x in img_list]
        img_list = [x for x in img_list if re.search("\.gitignore", x) is None]
        print(img_list)
        shuffle(img_list)
        img_list_sum.extend(img_list)
    shuffle(img_list_sum)
    train_length = int(len(img_list_sum) * 0.7)
    with open("./train.txt", 'a') as train_file:
        for index in range(0, train_length, 1):
            train_file.write(img_list_sum[index])
            train_file.write("\n")
    with open("./valid.txt", 'a') as valid_file:
        for index in range(train_length, len(img_list_sum), 1):
            valid_file.write(img_list_sum[index])
            valid_file.write("\n")


if __name__ == '__main__':
    main()
