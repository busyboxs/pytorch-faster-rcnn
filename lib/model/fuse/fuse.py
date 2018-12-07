import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pdb


class _L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(_L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class _Phase_concat(nn.Module):
    """concat feature with same resolution
    """
    def __init__(self, in_ch, reduce_dim=256):
        super(_Phase_concat, self).__init__()
        self.in_ch = in_ch
        self.reduce_dim = reduce_dim

        self.reduce256 = nn.Conv2d(self.in_ch, self.reduce_dim, kernel_size=1)

    def forward(self, x1, x2):
        x1 = F.relu(self.reduce256(x1), inplace=True)
        x2 = F.relu(self.reduce256(x2), inplace=True)

        x = torch.cat([x1, x2], dim=1)
        return x


class _Up_concat(nn.Module):
    """up low resolution feature map and concat with 
    high reolution feature map
    """
    def __init__(self, in_ch=512, out_ch=512, reduce_dim=256):
        super(_Up_concat, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.reduce_dim = reduce_dim

        self.reduce256 = nn.Conv2d(self.in_ch, self.reduce_dim, kernel_size=1)
        self.up = nn.ConvTranspose2d(reduce_dim, reduce_dim, 4, stride=2, bias=False)

        # self.L2Norm = _L2Norm(self.reduce_dim, 20)

        self.reduce512 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1_red = F.relu(self.reduce256(x1), inplace=True)
        x1_up = self.up(x1_red)
        x2_red = F.relu(self.reduce256(x2), inplace=True)

        # input is NCHW
        diffH = x2_red.size()[2] - x1_up.size()[2]
        diffW = x2_red.size()[3] - x1_up.size()[3]

        x1_up_pad = F.pad(x1_up, (diffW // 2, diffW - diffW // 2,
                                  diffH // 2, diffH - diffH // 2))

        # x1_up_l = self.L2Norm(x1_up_pad)
        # x2_l = self.L2Norm(x2_red)
        x = torch.cat([x2_red, x1_up_pad], dim=1)
        x_out = F.relu(self.reduce512(x), inplace=True)

        return x_out
