import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

from tools.online_negative_mining import OnlineHardNegativeMiningTripletLoss
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
    if args.train.test_monitor:
        # test_datasets = load_test_datasets(args.dataset_kwargs.get('data_dir'), args.debug)
        test_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
                train=False,
                **args.dataset_kwargs),
            shuffle=False,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    base_lr = args.train.base_lr*args.train.batch_size/256
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=base_lr,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256,
        args.train.num_epochs, base_lr, args.train.final_lr*args.train.batch_size/256,
        args.train.steps_per_epoch,
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, mode='Random', device=device)

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    val_accuracy = 0
    test_accuracy = 0
    min_accuracy = 200
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            model.zero_grad()
            # p1, p2, z1, z2 = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            # loss = criterion(p1, z2) + criterion(p2, z1)
            # data_dict = {'loss': loss}
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            data_dict['loss'] = data_dict['loss'].mean() # ddp
            loss = data_dict['loss']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

            if (idx + 1) % args.train.steps_per_epoch == 0:
                break

        # if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
        #     accuracy = knn_monitor(model.module.backbone, val_loader, test_loader, device, k=min(args.train.knn_k, len(val_loader.dataset)), hide_progress=args.hide_progress)

        epoch_dict = {"epoch": epoch}

        if args.train.val_monitor and epoch % args.train.val_interval == 0:
            model.eval()
            val_accuracy = evaluate_validation(model.module.backbone, val_loader, device)
            epoch_dict["val_accuracy"] = val_accuracy
        if args.train.test_monitor and epoch % args.train.test_interval == 0:
            model.eval()
            # test_accuracy = evaluate_test(model.module.backbone, test_datasets, device)
            test_accuracy = evaluate_validation(model.module.backbone, test_loader, device)
            epoch_dict["test_accuracy"] = test_accuracy

        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        if val_accuracy < min_accuracy:
            min_accuracy = val_accuracy
            # Save checkpoint
            model_path = os.path.join(args.log_dir, 'checkpoints',
                                      f"{args.name}_{val_accuracy}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    print('Device:', args.device)
    print(f"Using {torch.cuda.device_count()} GPUs")
    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














