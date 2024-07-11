'''
@File    :   make_data_txt.py
@Time    :   2024/07/11 15:03:20
@Author  :   SpringC
@Version :   1.0
@Contact :   huangw@xy.gabjoy.com
@Desc    :   经过统计{'car', 'bus', 'bicycle', 'boat', 'train', 'person', 'ambulance', 'motorcycle', 'truck'}

            只取车相关：排除boat， person
            1:patch_img不打乱,顺序截取train val test 7:2:!
            2:patch_pad_img不打乱
            3:patch_img打乱
            4:patch_pad_img打乱
            5:1和2的合并
            6:3和4的合并
'''



import os
import shutil
from springc_utils import *
import random
all_types = set()
txt_root = "./assets"


def read_txt_all_files(txt_path):
    patch_img = []
    patch_pad_img = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            if line[0] in ['boat', 'person']:continue # 不是车辆，过滤
            patch_img.append(f"{line[1]} {line[3]}")
            patch_pad_img.append(f"{line[2]} {line[3]}")
    print(f"patch_img共有: {len(patch_img)}")
    print(f"patch_pad_img共有: {len(patch_pad_img)}")
    return patch_img, patch_pad_img

def split_data(patch_img, patch_pad_img):#返回 随机和不随机两种划分
    n1, n2 = len(patch_img), len(patch_pad_img)
    train_num1, val_num1 = int(n1*0.7), int(n1*0.9)
    train_num2, val_num2 = int(n2*0.7), int(n2*0.9)
    train_data1, val_data1, test_data1 = patch_img[:train_num1], patch_img[train_num1:val_num1], patch_img[val_num1:]
    train_data2, val_data2, test_data2 = patch_pad_img[:train_num2], patch_pad_img[train_num2:val_num2], patch_pad_img[val_num2:]

    random.shuffle(patch_img)
    random.shuffle(patch_pad_img)
    train_data3, val_data3, test_data3 = patch_img[:train_num1], patch_img[train_num1:val_num1], patch_img[val_num1:]
    train_data4, val_data4, test_data4 = patch_pad_img[:train_num2], patch_pad_img[train_num2:val_num2], patch_pad_img[val_num2:]
    return (train_data1, val_data1, test_data1), (train_data2, val_data2, test_data2), (train_data3, val_data3, test_data3), (train_data4, val_data4, test_data4)

def main():
    txt_path = r"/mnt/hd1/springc/ImageData/public/nuscenes/patch_nuscene/patch_datas.txt"
    patch_img, patch_pad_img = read_txt_all_files(txt_path)
    datas = split_data(patch_img, patch_pad_img)
    sv_txt = ["train", 'val', 'test']
    for i, data in enumerate(datas):
        save_trainvaltest_txt(data, sv_txt, i+1)
    save_trainvaltest_txt(datas[0], sv_txt, 5)
    save_trainvaltest_txt(datas[1], sv_txt, 5)
    save_trainvaltest_txt(datas[2], sv_txt, 6)
    save_trainvaltest_txt(datas[3], sv_txt, 6)


def save_trainvaltest_txt(data, sv_txt, i):
    for tvt, s_p in zip(data, sv_txt):
        sv_txt_path = os.path.join(txt_root, f"{s_p}_{i}.txt")
        wf = open(sv_txt_path, 'a', encoding='utf-8')
        for t in tvt:
            wf.write(f"{t}\n")
        wf.close()

if __name__ == '__main__':
    main()
