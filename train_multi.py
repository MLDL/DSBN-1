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
    parser.add_argument('--batch-size', help='batch_size', default=40, type=int)
    parser.add_argument('--optimizer', help='[Adam/SGD]', default='Adam', type=str)
    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', help='learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight-decay', help='weight decay', default=0.0, type=float)

    args = parser.parse_args(args=args, namespace=namespace)
    return args 

def main():
    args =  parse_args()

    #data loading 


    #model loading 
    model = get_model(args.model_name, args.num_classes, args.in_feature)
    model.train(True)
    model = model.cuda(args.gpu)
