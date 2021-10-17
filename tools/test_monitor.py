import glob
import os

import numpy as np
import torch

from datasets import VisnirDataset
from tools.image import normalize_image
from tqdm import tqdm
import torch.nn.functional as F


def load_test_datasets(data_dir, debug):
    test_dir = os.path.join(data_dir, 'test\\')
    file_list = glob.glob(test_dir + "*.hdf5")
    test_data = dict()
    for f in file_list:
        path, dataset_name = os.path.split(f)
        dataset_name = os.path.splitext(dataset_name)[0]

        data = VisnirDataset.read_hdf5_data(f)

        x = data['Data'].astype(np.float32)
        test_labels = torch.from_numpy(np.squeeze(data['Labels']))
        del data

        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()

        x = normalize_image(x)
        x = x.repeat(3, axis=1)
        x = torch.from_numpy(x)

        test_data[dataset_name] = dict()
        test_data[dataset_name]['Data'] = x
        test_data[dataset_name]['Labels'] = test_labels
        del x
        if debug:
            break

    return test_data


def evaluate_test(net, test_data, device, step_size=800):
    samples_amount = 0
    total_test_err = 0
    for dataset_name in test_data:
        dataset = test_data[dataset_name]
        emb = evaluate_network(net, dataset['Data'][:, :, :, :, 0], dataset['Data'][:, :, :, :, 1], device, step_size)
        dist = np.power(emb['Emb1'] - emb['Emb2'], 2).sum(1)
        dataset['TestError'] = FPR95Accuracy(dist, dataset['Labels']) * 100
        del dist
        total_test_err += dataset['TestError'] * dataset['Data'].shape[0]
        samples_amount += dataset['Data'].shape[0]
    total_test_err /= samples_amount

    del emb
    return total_test_err


def FPR95Accuracy(dist_mat, labels):
    pos_indices = np.squeeze(np.asarray(np.where(labels == 1)))
    neg_indices = np.squeeze(np.asarray(np.where(labels == 0)))

    neg_dists = dist_mat[neg_indices]
    pos_dists = np.sort(dist_mat[pos_indices])

    thresh = pos_dists[int(0.95 * pos_dists.shape[0])]

    fp = sum(neg_dists < thresh)

    return fp / float(neg_dists.shape[0])


def evaluate_network(net, data1, data2, device, step_size=800):
    with torch.no_grad():

        for k in range(0, data1.shape[0], step_size):

            a = data1[k:(k + step_size), :, :, :]
            b = data2[k:(k + step_size), :, :, :]

            a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
            x1, x2 = net(a), net(b)
            x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)
            if k == 0:
                emb = dict()
                emb['Emb1'] = np.zeros(tuple([data1.shape[0]]) + tuple(x1.shape[1:]), dtype=np.float32)
                emb['Emb2'] = np.zeros(tuple([data1.shape[0]]) + tuple(x2.shape[1:]), dtype=np.float32)

            emb['Emb1'][k:(k + step_size)] = x1.cpu()
            emb['Emb2'][k:(k + step_size)] = x2.cpu()

    return emb


def evaluate_validation(net, val_loader, device):
    for ((images1, images2), labels) in tqdm(val_loader, desc='Validation', leave=False, disable=True):
        val_emb = evaluate_network(net, images1, images2, device)
        dist = np.power(val_emb['Emb1'] - val_emb['Emb2'], 2).sum(1)
        val_err = FPR95Accuracy(dist, labels) * 100
        return val_err