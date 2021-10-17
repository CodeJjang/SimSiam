import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.utils.data import WeightedRandomSampler
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
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            test=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_datasets = load_test_datasets(args.dataset_kwargs.get('data_dir'), args.debug)

    # define model
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model = torch.nn.DataParallel(model)

    model.eval()
    val_accuracy = evaluate_validation(model.module.backbone, val_loader, device)
    test_accuracy = evaluate_test(model.module.backbone, test_datasets, device)

    print('Val FPR95', val_accuracy)
    print('Test FPR95', test_accuracy)

if __name__ == "__main__":
    args = get_args()

    print('Device:', args.device)
    print(f"Using {torch.cuda.device_count()} GPUs")
    test(device=args.device, args=args)














