from torch import nn

class _DomainSpecificBatchNorm(nn.Module):
    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        self.bns = nn.ModuleList(
                [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _  in  range(num_classes)])

    def forward(self, x, domain_label):
        bn =  self.bns[domain_label[0]]
        return  bn(x), domain_label

class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):

