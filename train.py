# encoding = utf-8

import argparse
import datetime
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from models import build_model  # 假设有一个模型构建函数
from datasets import build_dataset
from engine import train_one_epoch  # 训练逻辑函数
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help = False)
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--lr_backbone', default = 1e-5, type = float)
    parser.add_argument('--batch_size', default = 2, type = int)
    parser.add_argument('--weight_decay', default = 1e-4, type = float)
    parser.add_argument('--epochs', default = 100, type = int)
    parser.add_argument('--lr_drop', default = 200, type = int)
    parser.add_argument('--clip_max_norm', default = 0.1, type = float,
                        help = 'gradient clipping max norm')
    
    # load pre trained model
    parser.add_argument('--pretrained', default = None, type = str, 
                help = 'Path to pretrained model')

    # Model parameters
    parser.add_argument('--frozen_weights', type = str, default = None,
                        help = 'Path to the pretrained model.'
                        'If set, only the mask head will be trained')
    # * Backbone
    parser.add_argument('--backbone', default = 'resnet50', type = str,
                        help = 'Name of the convolutional backbone to use')
    parser.add_argument('--dilation', action = 'store_true',
                        help = 'If true, we replace stride with dilation in the last convolutional block (DC5)')
    parser.add_argument('--position_embedding', default = 'sine', type = str, 
                        choices=('sine', 'learned'),
                        help = 'Type of positional embedding to use on top of the image features')

    # * Transformer
    parser.add_argument('--enc_layers', default = 6, type = int,
                        help = 'Number of encoding layers in the transformer')
    parser.add_argument('--dec_layers', default = 6, type = int,
                        help = 'Number of decoding layers in the transformer')
    parser.add_argument('--dim_feedforward', default = 2048, type = int,
                        help = 'Intermediate size of the feedforward layers in the transformer blocks')
    parser.add_argument('--hidden_dim', default = 256, type = int,
                        help = 'Size of the embeddings (dimension of the transformer)')
    parser.add_argument('--dropout', default = 0.1, type = float,
                        help = 'Dropout applied in the transformer')
    parser.add_argument('--nheads', default = 8, type = int,
                        help = 'Number of attention heads inside the transformer\'s attentions')
    parser.add_argument('--num_queries', default = 100, type = int,
                        help = 'Number of query slots')
    parser.add_argument('--pre_norm', action = 'store_true')

    # * Segmentation
    parser.add_argument('--masks', action = 'store_true',
                        help = 'Train segmentation head if the flag is provided')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action = 'store_false',
                        help = 'Disables auxiliary decoding losses (loss at each layer)')
    # * Matcher
    parser.add_argument('--set_cost_class', default = 1, type = float,
                        help = 'Class coefficient in the matching cost')
    parser.add_argument('--set_cost_bbox', default = 5, type = float,
                        help = 'L1 box coefficient in the matching cost')
    parser.add_argument('--set_cost_giou', default = 2, type = float,
                        help = 'giou box coefficient in the matching cost')
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default = 1, type = float)
    parser.add_argument('--dice_loss_coef', default = 1, type = float)
    parser.add_argument('--bbox_loss_coef', default = 5, type = float)
    parser.add_argument('--giou_loss_coef', default = 2, type = float)
    parser.add_argument('--eos_coef', default = 0.1, type = float,
                        help = 'Relative classification weight of the no-object class')

    # dataset parameters
    parser.add_argument('--dataset_file', default = 'default')
    parser.add_argument('--coco_path', type = str)
    parser.add_argument('--coco_panoptic_path', type = str)
    parser.add_argument('--remove_difficult', action = 'store_true')

    parser.add_argument('--output_dir', default = '',
                        help = 'path where to save, empty for no saving')
    parser.add_argument('--device', default = 'cpu',
                        help = 'device to use for training / testing')
    parser.add_argument('--seed', default = 42, type = int)
    parser.add_argument('--resume', default = '', help = 'resume from checkpoint')
    parser.add_argument('--start_epoch', default = 0, type = int, metavar='N',
                        help = 'start epoch')
    parser.add_argument('--eval', action = 'store_true')
    parser.add_argument('--num_workers', default = 16, type = int)

    # distributed training parameters
    parser.add_argument('--world_size', default = 1, type = int,
                        help = 'number of distributed processes')
    parser.add_argument('--dist_url', default = 'env://', 
                    help = 'url used to set up distributed training')

    return parser

def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 数据预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("Loading dataset...")
    dataset_train = build_dataset(image_set = 'train', args = args)
    dataset_val = build_dataset(image_set = 'val', args = args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last = True)

    data_loader_train = DataLoader(dataset_train, batch_sampler = batch_sampler_train,
                                collate_fn = utils.collate_fn, num_workers = args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler = sampler_val,
                                drop_last = False, collate_fn = utils.collate_fn, 
                                num_workers = args.num_workers)

    # 构建模型
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 加载预训练模型
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location = device)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No pretrained model specified or found, training from scratch.")
        exit(0)

    # 设置优化器
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr / 10},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 训练循环
    print("Starting training...")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, 
                                optimizer, device, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Time: {time.time() - start_time:.2f}s")

        # 每10个 epoch 保存一次模型
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training completed in {total_time_str}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)