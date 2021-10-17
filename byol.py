import torch
from torchvision import models
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
from models.byol_pytorch import BYOL
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime

from tools.test_monitor import load_test_datasets, evaluate_test, evaluate_validation

def main(device, args):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
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
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    resnet = models.resnet50(pretrained=True)
    resnet = resnet.to(device)
    # resnet = torch.nn.DataParallel(resnet)

    learner = BYOL(
        resnet,
        image_size = 32,
        hidden_layer = 'avgpool',
        augment_fn=lambda x: x
    )

    opt = get_optimizer(
        args.train.optimizer.name, learner,
        lr=args.train.base_lr * args.train.batch_size / 256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    # lr_scheduler = LR_Scheduler(
    #     opt,
    #     args.train.warmup_epochs, args.train.warmup_lr * args.train.batch_size / 256,
    #     args.train.num_epochs, args.train.base_lr * args.train.batch_size / 256,
    #                               args.train.final_lr * args.train.batch_size / 256,
    #     args.train.steps_per_epoch,
    #     constant_predictor_lr=True  # see the end of section 4.2 predictor
    # )

    def sample_unlabelled_images():
        return torch.randn(20, 3, 256, 256)

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        learner.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            opt.zero_grad()
            images1, images2 = images1.to(device, non_blocking=True), images2.to(device, non_blocking=True, )
            loss = learner(images1, images2)
            loss = loss.mean()
            data_dict = {'loss': loss.item()}
            loss.backward()
            opt.step()
            # lr_scheduler.step()
            learner.update_moving_average() # update moving average of target encoder

            # data_dict.update({'lr': lr_scheduler.get_lr()})
            data_dict.update({'lr': opt.param_groups[0]['lr']})
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            accuracy = knn_monitor(learner.online_encoder, val_loader, test_loader, device, k=min(args.train.knn_k, len(val_loader.dataset)), hide_progress=args.hide_progress)

        if 'val_monitor' in vars(args.train) and args.train.val_monitor and epoch % args.train.val_interval == 0:
            learner.eval()
            accuracy = evaluate_validation(learner.online_encoder, val_loader, device)

        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    # save your improved network
    # torch.save(resnet.state_dict(), './improved-net.pt')

if __name__ == "__main__":
    args = get_args()

    print('Device:', args.device)
    print(f"Using {torch.cuda.device_count()} GPUs")
    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
