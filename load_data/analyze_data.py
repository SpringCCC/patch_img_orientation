'''
@File    :   analyze_data.py
@Time    :   2024/07/11 18:17:47
@Author  :   SpringC
@Version :   1.0
@Contact :   huangw@xy.gabjoy.com
@Desc    :   分析图像的size
'''


from springc_utils import *
from PIL import Image
import random
import matplotlib.pyplot as plt

def readtxt(txt_path):
    imgs_path = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            if line[0] in ['person', 'boat', 'train']:continue
            imgs_path.append(line[1])
    return imgs_path

def analyze(txt_path, num=1000):
    imgs_path = readtxt(txt_path)
    all_size = []
    his_max = []
    his_min = []
    rate = []
    for img_path in imgs_path:
        img = Image.open(img_path)
        w, h = img.size
        all_size.append([w, h])
        his_max.append(max(w, h))
        his_min.append(min(w, h))
        rate.append(max(w/h, h/w))
    sample_size = random.sample(all_size, num)
    sample_size = np.asarray(sample_size)
    widths = sample_size[:, 0]
    heights = sample_size[:, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Image Width vs Height')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True)
    plt.savefig(f"analyze_wh_{num}.png")

    #分桶
    counts, bin_edges = np.histogram(np.asarray(his_max), bins=np.arange(0, 1601, 10))  
    plt.figure(figsize=(10, 6))  # 设置图形的大小  
    plt.bar(bin_edges[:-1], counts, width=10, color='blue', edgecolor='black')  # 绘制条形图  
    plt.xlabel('Value Range')  # 设置x轴标签  
    plt.ylabel('Counts')  # 设置y轴标签  
    plt.title('Max Values of wh')  # 设置图形标题  
    plt.xticks(np.arange(0, 1601, 100))  # 设置x轴的刻度  
    plt.savefig(f"analyze_maxwh_{num}.png")

    counts, bin_edges = np.histogram(np.asarray(his_min), bins=np.arange(0, 1601, 10))  
    plt.figure(figsize=(10, 6))  # 设置图形的大小  
    plt.bar(bin_edges[:-1], counts, width=10, color='blue', edgecolor='black')  # 绘制条形图  
    plt.xlabel('Value Range')  # 设置x轴标签  
    plt.ylabel('Counts')  # 设置y轴标签  
    plt.title('Min Values of wh')  # 设置图形标题  
    plt.xticks(np.arange(0, 1601, 100))  # 设置x轴的刻度  
    plt.savefig(f"analyze_minwh_{num}.png")

    bins = np.linspace(1, 5, 50)
    plt.hist(np.asarray(rate), bins=bins, color='blue', edgecolor='black', alpha=0.7)  
    plt.xlabel('max(w/h, h/w) Value')  # x轴标签  
    plt.ylabel('Counts')  # y轴标签  
    plt.title('w/h or h/w bigger Values')  # 图表标题  
    # plt.xticks(np.arange(np.asarray(rate).min(), np.asarray(rate).max(), 100))
    plt.xticks(np.arange(1,5, 0.1))  # 刻度从 0 到 10，间隔为 1，并旋转 45 度以便阅读
    plt.savefig(f"analyze_max_w_divide_h_{num}.png")

    
if __name__ == "__main__":
    # root = r"/mnt/hd1/springc/ImageData/public/nuscenes/patch_nuscene/"
    # analyze(root, 100)

    txt_path = r"/mnt/hd1/springc/ImageData/public/nuscenes/patch_nuscene/patch_datas.txt"
    analyze(txt_path, 10000)
