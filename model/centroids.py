import torch
from torch import nn as nn

class Centroids(nn.Module):
    def __init__(self, feature_dim, num_classes, decay_const=0.3):
        super(Centroids, self).__init__()
        self.decay_const = decay_const
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))
        #torch.nn.Parameter : A kind of Tensor that is to be considered a module parameter 
        self.centroids.requires_grad =  False
        self.reset_parameters()

    def reset_parameters(self):
        self.centroids.data.zero_()

    def forward(self, x, y, y_mask=None):
        #torch.unique(*args, **kwargs): returns the unique elements of the input tensor 
        classes = torch.unique(y)
        current_centroids = []
        return current_centroids
