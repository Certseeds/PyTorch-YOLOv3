import os
from random import shuffle


def main():
    print(os.path.abspath("./"))
    this_before = "./../../"
    root_before = "../"
    repo_path = "barcodeartelab/images/JPEGImages/"
    path = this_before + root_before + repo_path
    img_list = os.listdir(path)
    img_list = [root_before + repo_path + x for x in img_list]
    print(img_list)
    shuffle(img_list)
    train_length = int(len(img_list) * 0.8)
    with open("./train.txt", 'r+') as train_file:
        for index in range(0, train_length, 1):
            train_file.write(img_list[index])
            train_file.write("\n")
    with open("./valid.txt", 'r+') as valid_file:
        for index in range(train_length, len(img_list), 1):
            valid_file.write(img_list[index])
            valid_file.write("\n")


if __name__ == '__main__':
    main()
