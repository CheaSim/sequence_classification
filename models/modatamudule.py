import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os
import random
from transformers import BertTokenizer

class MoDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MoDataModule, self).__init__()
        self.args = args
        self.name_zh = {'LX': '鲁迅', 'MY':'莫言' , 'QZS':'钱钟书' ,'WXB':'王小波' ,'ZAL':'张爱玲'} 
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    def setup(self, ):
        args = self.args
        sentences = [] # 片段
        target = [] # 作者
    
        # 定义lebel到数字的映射关系
        labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

        files = os.listdir(args.dataset)
        for file in files:
            if not os.path.isdir(file):
                f = open(args.dataset + "/" + file, 'r', encoding='UTF-8');  # 打开文件
                for index,line in enumerate(f.readlines()):
                    sentences.append(line)
                    target.append(labels[file[:-4]])


        temp = list(zip(sentences, target))
        random.shuffle(temp)
        self.data = temp

    def prepare_data(self, ):
        tokenizer = self.tokenizer

        X = tokenizer.batch_encode_plus(self.data[:,0], padding=True, )
        labels = torch.from_numpy(self.data[:,1])

        total_size = labels.size()[0]
        train_sz = int(total_size * 0.9)
        self.training_dataset = Dataset(X[0:train_sz], labels[0:train_sz])
        self.validation_dataset = Dataset(X[train_sz:], labels[train_sz:])

    def train_dataloader(self,) -> DataLoader:
        return DataLoader(self.training_dataset, batch_size = self.args.batch_size, shuffle=True, num_workers = 8)

    def val_dataloader(self, *args, **kwargs) :
        return DataLoader(self.validation_dataset, batch_size = self.args.batch_size, shuffle=True, num_workers = 8)




