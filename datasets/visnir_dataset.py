import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tools.image import normalize_image
import torch
import os
import glob


class VisnirDataset(Dataset):

    def __init__(self, data_dir, transform=None, train=True, test=False):
        if not test:
            data_dir = os.path.join(data_dir, 'train\\train.hdf5')
            self.init_trainval(data_dir, transform, train)
        else:
            self.init_test(data_dir, transform)

    def init_trainval(self, data_dir, transform, train):
        data = VisnirDataset.read_hdf5_data(data_dir)
        train_data = data['Data']
        train_labels = np.squeeze(data['Labels'])
        train_split = np.squeeze(data['Set'])
        del data

        if not train:
            val_indices = np.squeeze(np.asarray(np.where(train_split == 3)))

            # VALIDATION data
            val_labels = torch.from_numpy(train_labels[val_indices])
            val_data = np.squeeze(train_data[val_indices])

            data = val_data
            labels = val_labels
        else:
            train_indices = np.squeeze(np.asarray(np.where(train_split == 1)))
            data = np.squeeze(train_data[train_indices])
            labels = train_labels[train_indices]

        self.pos_indices = np.squeeze(np.asarray(np.where(labels == 1)))
        self.neg_indices = np.squeeze(np.asarray(np.where(labels == 0)))

        self.pos_amount = len(self.pos_indices)
        self.neg_amount = len(self.neg_indices)

        self.data = data
        self.labels = labels

        self.transform = transform

        self.data_height = self.data.shape[1]
        self.data_width = self.data.shape[2]

    def init_test(self, data_dir, transform):
        data_dict = VisnirDataset.read_hdf5_data(data_dir)
        data = np.squeeze(data_dict['Data']) # .astype(np.float32)
        labels = torch.from_numpy(np.squeeze(data_dict['Labels']))
        del data_dict

        self.pos_indices = np.squeeze(np.asarray(np.where(labels == 1)))
        self.neg_indices = np.squeeze(np.asarray(np.where(labels == 0)))

        self.pos_amount = len(self.pos_indices)
        self.neg_amount = len(self.neg_indices)

        self.data = data
        self.labels = labels

        self.transform = transform

        self.data_height = self.data.shape[1]
        self.data_width = self.data.shape[2]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Select pos pairs
        # pos_idx = np.random.randint(self.pos_amount)

        # pos_idx = self.pos_indices[pos_idx]
        # pos_images = self.data[pos_idx, :, :, :]
        images = self.data[index]

        im1 = images[..., 0, None].repeat(3, axis=2)
        im2 = images[..., 1, None].repeat(3, axis=2)

        im1 = Image.fromarray(im1)
        im2 = Image.fromarray(im2)

        if self.transform:
            im1, im2 = self.transform(im1, im2)

        label = self.labels[index]
        return [im1, im2], label

    @staticmethod
    def read_hdf5_data(fname):
        with h5py.File(fname, 'r') as f:

            keys = list(f.keys())

            if len(keys) == 1:
                data = f[keys[0]]
                res = np.squeeze(np.array(data[()]))
            else:
                i = 0
                res = dict()
                for v in keys:
                    res[v] = np.array(f[keys[i]])
                    i += 1

        return res
