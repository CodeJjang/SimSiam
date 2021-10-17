import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.utils.data import WeightedRandomSampler, ConcatDataset
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime

from tools.test_monitor import load_test_datasets, evaluate_test, evaluate_validation
import glob

def test(device, args):

    val_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_datasets = []
    file_list = glob.glob(os.path.join(args.data_dir, 'test\\') + "*.hdf5")
    for f in file_list:
        _, dataset_name = os.path.split(f)
        dataset_name = os.path.splitext(dataset_name)[0]
        dataset = get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            test=True,
            data_dir=f,
            dataset=args.dataset_kwargs['dataset'])
        setattr(dataset, 'name', dataset_name)
        test_datasets.append(dataset)
        if args.debug:
            break
    test_loader = torch.utils.data.DataLoader(
        dataset=ConcatDataset(test_datasets),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    # test_datasets = load_test_datasets(args.dataset_kwargs.get('data_dir'), args.debug)

    # define model
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model = torch.nn.DataParallel(model)

    model.eval()
    val_accuracy = evaluate_validation(model.module.backbone, val_loader, device)
    test_accuracy = evaluate_validation(model.module.backbone, test_loader, device)

    print('Val FPR95', val_accuracy)
    print('Test FPR95', test_accuracy)


if __name__ == "__main__":
    args = get_args()

    print('Device:', args.device)
    print(f"Using {torch.cuda.device_count()} GPUs")
    test(device=args.device, args=args)














