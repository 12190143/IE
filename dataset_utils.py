# coding=utf-8
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        if mode == 'train':
            self.labels = [torch.tensor(example.labels) for example in features]

    def __len__(self):
        return self.nums


class MyDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode):
        super(MyDataset, self).__init__(features, mode)

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data
