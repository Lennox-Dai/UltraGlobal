import torch
import torch.nn as nn
import torch.nn.functional as F
from model.CVNet_Rerank_model import CVNet_Rerank

def setup_model(model_depth, reduction_dim=2048, SupG={'gemp': True, 'sgem': True, 'rgem': True, 'relup': True, 'rerank': True, 'onemeval': False}):
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(model_depth, reduction_dim, SupG['relup'])
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model

class GeneralizedMeanPooling(nn.Module):

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)

class GlobalHead(nn.Module):
    def __init__(self, w_in, nc, pp=3):
        super(GlobalHead, self).__init__()
        self.fc = nn.Linear(w_in, nc, bias=True)
        self.pool = GeneralizedMeanPoolingP(norm=pp)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class sgem(nn.Module):

    def __init__(self, ps=10., infinity = True):
        super(sgem, self).__init__()
        self.ps = ps
        self.infinity = infinity
    def forward(self, x):

        # x = torch.stack(x,0)
        x = x.unsqueeze(0)

        if self.infinity:
            x = F.normalize(x, p=2, dim=-1) # 3 C
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma

        return x
    
class rgem(nn.Module):

    def __init__(self, pr=2.5, size = 5):
        super(rgem, self).__init__()
        self.pr = pr
        self.size = size
        self.lppool = nn.LPPool2d(self.pr, int(self.size), stride=1)
        self.pad = nn.ReflectionPad2d(int((self.size-1)//2.))
    def forward(self, x):
        nominater = (self.size**2) **(1./self.pr)
        x = 0.5*self.lppool(self.pad(x/nominater)) + 0.5*x
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, noise_std=0.01):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv1d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.noise_std = noise_std

    def forward(self, x, epoch):
        batch_size, C, T = x.size()

        noise = torch.randn_like(x) * (self.noise_std / (2**epoch))
        x_noisy = x + noise
        x_noisy = x
        g_x = self.g(x_noisy).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = self.bn(W_y + x_noisy)

        return z
    
# class gemp(nn.Module):

#     def __init__(self, p=4.6, eps=1e-8, m=2048):
#         super(gemp, self).__init__()
#         self.m = m
#         self.p = [p] * m
#         self.eps = [eps] * m

#     def forward(self, x):
#         pooled_features = []
#         for i in range(self.m):
#             x_clamped = x.clamp(self.eps[i]).pow(self.p[i])
#             pooled = torch.nn.functional.adaptive_avg_pool1d(x_clamped, 1).pow(1. / self.p[i])
#             pooled_features.append(pooled)

#         concatenated_features = torch.cat(pooled_features, dim=-1)

#         return concatenated_features

class gemp(nn.Module):
    def __init__(self, p=4.6, eps=1e-8, channel=2048, m=2048):
        super(gemp, self).__init__()
        self.m = m
        self.p = [p] * m
        self.eps = [eps] * m
        self.nonlocal_block = NonLocalBlock(channel, noise_std=0.001)

    def forward(self, x, epoch):
        pooled_features = []
        for i in range(self.m):
            x1 = self.nonlocal_block(x, epoch=epoch)
            x_clamped = x1.clamp(self.eps[i]).pow(self.p[i])
            pooled = torch.nn.functional.adaptive_avg_pool1d(x_clamped, 1).pow(1. / self.p[i])
            pooled_features.append(pooled)

        concatenated_features = torch.cat(pooled_features, dim=-1)

        return concatenated_features