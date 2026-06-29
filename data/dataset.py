import random
import numpy as np
from torch.utils.data.dataset import Dataset
from config import cfg

class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True, sample_prob=None):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len
        self.sample_prob = None
        if sample_prob is not None:
            sample_prob = np.asarray(sample_prob, dtype=np.float32)
            if sample_prob.shape[0] != self.db_num:
                raise ValueError("sample_prob length must match the number of datasets")
            if np.any(sample_prob < 0) or sample_prob.sum() <= 0:
                raise ValueError("sample_prob must be non-negative and have a positive sum")
            self.sample_prob = sample_prob / sample_prob.sum()

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.sample_prob is not None:
            db_idx = int(np.random.choice(np.arange(self.db_num), p=self.sample_prob))
            data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
        elif self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        inputs, targets, meta_info = self.dbs[db_idx][data_idx]
        if 'coco_hand_joint_img' not in targets:
            targets['coco_hand_joint_img'] = np.zeros((2, 21, 3), dtype=np.float32)
        if 'coco_hand_joint_trunc' not in targets:
            targets['coco_hand_joint_trunc'] = np.zeros((2, 21, 1), dtype=np.float32)
        return inputs, targets, meta_info
