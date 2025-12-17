import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from modules.radgraph_m import RadGraph
from utils.radgraph_download import RADGRAPH_MODEL_DIR


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args["image_dir"]
        self.ann_path = args["ann_path"]
        self.max_seq_length = args["max_seq_length"]
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        self.limit_val_length = None
        self.limit_test_length = None

        # print(len(self.examples))

        if args["dataset_name"] == "mimic_cxr_mini":
            self.limit_val_length = 237
            self.limit_test_length = 474

        #平衡训练集、测试集、验证集的大小
        if self.split == 'val':
            self.examples = self.ann[self.split]
        elif self.split == 'test':
            self.examples = self.ann[self.split]
        else:
            # 训练集保持完整
            self.examples = self.ann[self.split]


        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            # self.examples[i]['re'] = tokenizer(self.examples[i]['report'],out_type=False)[:self.max_seq_length]

    def __len__(self):
        return len(self.examples)

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # report_ori = example['re']
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class MimicMiniMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            # image_2 = self.transform(image_2)
        # image = torch.stack((image_1, image_2), 0)
        image = image_1
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # report_ori = example['re']
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class MimicSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
        image = image_1
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample