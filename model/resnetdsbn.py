import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def resnet50dsbn(**kwargs):
    model = DSBNResNet(Bottleneck, [3,4,6,3], **kwargs)
    return model 

class DSBNResNEt(nn.Module):
    def __init__(self, block, layers, in_features=256, num_classes=1000, num_domains=2):
        super(DSBNResNet, self).__init__()
        self.inplanes = 64
        self.in_features = in_features
        self.num_domains = num_domains
        self.num_classes = num_classes
        
