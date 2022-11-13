
import argparse

import h5py
import numpy as np


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo_labels_1', type=str, required=True)
    parser.add_argument('--pseudo_labels_2', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    pl_1 = read_hdf5_data(args.pseudo_labels_1)
    pl_2 = read_hdf5_data(args.pseudo_labels_2)
    out_path = args.out
    data = np.concatenate([pl_1['Data'], pl_2['Data']])
    labels = np.concatenate([pl_1['Labels'], pl_2['Labels']])
    set = np.concatenate([pl_1['Set'], pl_2['Set']])
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('Data', data=data)
        f.create_dataset('Labels', data=labels)
        f.create_dataset('Set', data=set)













