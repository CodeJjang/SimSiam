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
from models.backbones.multiscale_transformer_encoder import MultiscaleTransformerEncoder
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime

from tools.test_monitor import load_test_datasets, evaluate_test, evaluate_validation, evaluate_network
import glob
import h5py
import re
from pathlib import Path



def generate_pseudo_labels(device, args):
    if not args.cycle_name:
        print('Cycle name missing')
        return

    # define model
    state_dict = torch.load(args.checkpoint)['state_dict']
    if 'query' in state_dict.keys():
        model = MultiscaleTransformerEncoder(0.5).to(device)
        model_name = 'mt_encoder'
        channels = 1
    else:
        model = get_model(args.model).to(device).module.backbone
        model_name = 'cnn_backbone'
        channels = 3
    try:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    except Exception as e:
        print('Caught incorrect state dict, trying to resolve...')
        try_load_cnn_state_dict(model, torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict'])
    model = torch.nn.DataParallel(model)
    model.eval()
    args.dataset_kwargs['channels'] = channels
    args.aug_kwargs['channels'] = channels
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=True,
            shuffle=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.val.batch_size,
        **args.dataloader_kwargs
    )
    val_data = get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs)
    print(datetime.now(), 'Generating embeddings...')
    rgb_embeddings, rgb_images, nir_embeddings, nir_images = gen_embeddings(model, train_loader, device, model_name=model_name)
    print(datetime.now(), 'Finished generating embeddings')

    sim_matrix = np.dot(rgb_embeddings, nir_embeddings.transpose(1, 0))
    nir_most_similar_rgb_indices = np.argmax(sim_matrix, axis=0)
    rgb_most_similar_nir_indices = np.argmax(sim_matrix, axis=1)
    if args.sim == 'mutual_agreement':
        rgb_mutual_sim_indices = np.where(nir_most_similar_rgb_indices[rgb_most_similar_nir_indices] == np.arange(len(rgb_most_similar_nir_indices)))[0]
        sim = sim_matrix[rgb_mutual_sim_indices, rgb_most_similar_nir_indices[rgb_mutual_sim_indices]]
    elif args.sim == 'closest_pairs':
        sim = sim_matrix[np.arange(len(rgb_most_similar_nir_indices)), rgb_most_similar_nir_indices]
    else:
        raise Exception('Undefined similarity method', args.sim)
    topk = 10000
    most_similar_indices = np.argsort(sim)[-topk:]
    rgb_indices = rgb_most_similar_nir_indices[most_similar_indices]
    nir_indices = rgb_most_similar_nir_indices[rgb_indices]
    top_rgb_images = rgb_images[rgb_indices]
    top_nir_images = nir_images[nir_indices]
    top_rgb_images = top_rgb_images[:, 0][:, np.newaxis]
    top_nir_images = top_nir_images[:, 0][:, np.newaxis]
    train_data = np.hstack([top_rgb_images, top_nir_images]).transpose(0, 2, 3, 1)[:, np.newaxis]
    data = np.vstack([train_data, val_data.data[:, np.newaxis]])
    labels = np.concatenate([np.ones(len(train_data)), val_data.labels])
    set = np.concatenate([np.ones(len(train_data)), np.ones(len(val_data)) * 3])
    trainval_name = 'pseudo_labels_trainval_{}_{}_top_{}.hdf5'.format(model_name, args.sim, topk)
    save_path = os.path.join(args.dataset_kwargs['data_dir'], 'train', 'pseudo_labels', args.cycle_name, trainval_name)
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('Data', data=data)
        f.create_dataset('Labels', data=labels)
        f.create_dataset('Set', data=set)

    print(datetime.now(), 'Finished pseudo labeling')


def gen_embeddings(net, data_loader, device, adain=False, model_name='cnn_backbone'):
    rgb_embeddings = []
    nir_embeddings = []
    rgb_images = []
    nir_images = []
    for ((images1, images2), labels) in tqdm(data_loader, leave=False, disable=True):
        val_emb = evaluate_network(net, images1, images2, device, step_size=len(labels), adain=adain, model_name=model_name)
        rgb_embeddings.append(val_emb['Emb1'])
        nir_embeddings.append(val_emb['Emb2'])
        rgb_images.append(images1)
        nir_images.append(images2)
    rgb_embeddings = np.concatenate(rgb_embeddings)
    nir_embeddings = np.concatenate(nir_embeddings)
    rgb_images = np.concatenate(rgb_images)
    nir_images = np.concatenate(nir_images)

    shuffle_indices = np.random.choice(len(rgb_embeddings), size=len(rgb_embeddings), replace=False)
    rgb_embeddings = rgb_embeddings[shuffle_indices]
    rgb_images = rgb_images[shuffle_indices]
    nir_embeddings = nir_embeddings[shuffle_indices]
    nir_images = nir_images[shuffle_indices]
    return rgb_embeddings, rgb_images, nir_embeddings, nir_images


def try_load_cnn_state_dict(net, state_dict):
    state_dict = {k.split('backbone_cnn.')[1]: v for k, v in state_dict.items() if 'backbone_cnn' in k}
    new_state_dict = {}
    new_state_dict['block.0.weight'] = np.repeat(state_dict['pre_block.0.weight'], 3, axis=1)
    new_state_dict['block.1.running_mean'] = state_dict['pre_block.1.running_mean']
    new_state_dict['block.1.running_var'] = state_dict['pre_block.1.running_var']
    new_state_dict['block.1.num_batches_tracked'] = state_dict['pre_block.1.num_batches_tracked']

    blocks = list(set([int(re.findall(r'\d+', k)[0]) for k in state_dict.keys()]))
    block_pairs = list(zip(blocks[::2], blocks[1::2]))
    for block1, block2 in block_pairs:
        new_state_dict[f'block.{block1 + 3}.weight'] = state_dict[f'block.{block1}.weight']
        new_state_dict[f'block.{block2 + 3}.running_mean'] = state_dict[f'block.{block2}.running_mean']
        new_state_dict[f'block.{block2 + 3}.running_var'] = state_dict[f'block.{block2}.running_var']
        new_state_dict[f'block.{block2 + 3}.num_batches_tracked'] = state_dict[
            f'block.{block2}.num_batches_tracked']
    net.backbone.load_state_dict(new_state_dict, strict=True)


def try_load_mt_state_dict(net, state_dict):
    state_dict = {k.split('backbone_cnn.')[1]: v for k, v in state_dict.items() if 'backbone_cnn' in k}
    new_state_dict = {}
    new_state_dict['block.0.weight'] = np.repeat(state_dict['pre_block.0.weight'], 3, axis=1)
    new_state_dict['block.1.running_mean'] = state_dict['pre_block.1.running_mean']
    new_state_dict['block.1.running_var'] = state_dict['pre_block.1.running_var']
    new_state_dict['block.1.num_batches_tracked'] = state_dict['pre_block.1.num_batches_tracked']

    blocks = list(set([int(re.findall(r'\d+', k)[0]) for k in state_dict.keys()]))
    block_pairs = list(zip(blocks[::2], blocks[1::2]))
    for block1, block2 in block_pairs:
        new_state_dict[f'block.{block1 + 3}.weight'] = state_dict[f'block.{block1}.weight']
        new_state_dict[f'block.{block2 + 3}.running_mean'] = state_dict[f'block.{block2}.running_mean']
        new_state_dict[f'block.{block2 + 3}.running_var'] = state_dict[f'block.{block2}.running_var']
        new_state_dict[f'block.{block2 + 3}.num_batches_tracked'] = state_dict[
            f'block.{block2}.num_batches_tracked']
    net.backbone.load_state_dict(new_state_dict, strict=True)


if __name__ == "__main__":
    args = get_args()

    print('Device:', args.device)
    print(f"Using {torch.cuda.device_count()} GPUs")
    generate_pseudo_labels(device=args.device, args=args)














