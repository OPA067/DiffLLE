from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split
        if split == 'train':
            gt_dir = 'target'
            input_dir = 'input'
        else:
            gt_dir = 'target'
            input_dir = 'input'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        img_LR = Image.open(self.sr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        img_HR = Image.open(self.hr_path[index]).convert("RGB")

        [img_LR, img_SR, img_HR] = Util.transform_augment([img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))

        return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}

