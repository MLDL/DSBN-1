from torch import nn

class _DomainSpecificBatchNorm(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1):
        super(_DomainSpecificBatchNorm, self).__init__()

    def forward(self, x, domain_label):

