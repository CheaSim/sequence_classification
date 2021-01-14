import os
import random
import pandas as pd
import csv

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者
    
    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))

path = './dataset'

temp = load_data(path)
random.shuffle(temp)

for idx, a in enumerate(temp):
    t = [None,None]
    t[0] = a[0].replace('\n', '')
    t[1] = a[1]
    temp[idx] = t



sz = int(len(temp) * 0.95)
train_ds = temp[:sz]
val_ds = temp[sz:]



def save_tsv(file_name, data):
    with open(file_name, 'w', encoding='UTF-8') as writer:
        tsv_w = csv.writer(writer, delimiter='\t')
        tsv_w.writerow(['sentence', 'label'])  # 单行写入
        tsv_w.writerows(data)  # 多行写入

save_tsv('./data/train.tsv', train_ds)
save_tsv('./data/test.tsv', val_ds)
save_tsv('./data/dev.tsv', val_ds)

