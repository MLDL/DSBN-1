import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.train_utils import init_weights 

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2m inplace=True)

