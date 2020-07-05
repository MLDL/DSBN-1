from dataset.factory import get_dataset
from model.factory import get_model
from discriminator.factory import get_discriminator

from model.centroids import Centroids 
from torch.utils import data 

import torch.nn.functional as  F
import torch.optim as optim 
import os
import torch
import torch.nn as nn
import numpy as np

def parse_args(args=None, namespace=None):
    """
    parse input arguments
    """
    parser  = argparse.ArgumentParser()

    parser.add_argument('--model-name', default='resnet50dsbn', type=str)
    parser.add_argument('--exp-setting', help='exp setting[digits, office, imageclef, visda]', default='office',
                        type=str)
    parser.add_argument('--init-model-path', help='init model path', default='', type=str)
    parser.add_argument('--save-dir', help='directory to save models', default='output/office_default', type=str)

    parser.add_argument('--num-classes', default=0, type=int)
    parser.add_argument('--in-features', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument('--optimizer', help='[Adam/SGD]', default='Adam', type=str)
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    args = parser.parse_args(args=args, namespace=namespace)
    return args 

def main():
    args =  parse_args()
    args.dsbn = True
    torch.cuda.set_device(args.gpu)
    
    #if args.exp_setting == 'office'
    args.num_classes = 31
    
    num_classes = args.num_classes
    in_features = num_classes
    num_domains = len(args.source_datasets)+len(args.target_datasets)

    #data loading 
     source_train_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, source_name, 'train', args.jitter))
                             for source_name in args.source_datasets]
    target_train_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, target_name, 'train', args.jitter))
                             for target_name in args.target_datasets]

    if args.merge_sources:
        for i in range(len(source_train_datasets)):
            if i == 0:
                merged_source_train_datasets = source_train_datasets[i]
            else:
                # concatenate dataset
                merged_source_train_datasets = merged_source_train_datasets + source_train_datasets[i]
        source_train_datasets = [merged_source_train_datasets]

    # dataloader
    source_train_dataloaders = [data.DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
                                for source_train_dataset in source_train_datasets]
    target_train_dataloaders = [data.DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
                                for target_train_dataset in target_train_datasets]

    source_train_dataloader_iters = [enumerate(source_train_dataloader) for source_train_dataloader in
                                     source_train_dataloaders]
    target_train_dataloader_iters = [enumerate(target_train_dataloader) for target_train_dataloader in
                                     target_train_dataloaders]

    # validation dataloader
    target_val_datasets = [get_dataset("{}_{}_{}_{}".format(args.model_name, target_name, 'val', args.jitter))
                           for target_name in args.target_datasets]
    target_val_dataloaders = [data.DataLoader(target_val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
                              for target_val_dataset in target_val_datasets]

    #model loading 
    model = get_model(args.model_name, args.num_classes, args.in_feature)
    model.train(True)
    model = model.cuda(args.gpu)

     params = get_optimizer_params(model, args.learning_rate, weight_decay=args.weight_decay,double_bias_lr=args.double_bias_lr, base_weight_factor=args.base_weight_factor)
    
    #adversarial loss

    #semantic loss 

