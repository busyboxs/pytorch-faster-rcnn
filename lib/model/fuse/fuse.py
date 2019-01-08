import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
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


class _ScaledL2Norm(nn.Module):
    def __init__(self, in_channels, initial_scale):
        super(_ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        return x * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)


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
    def __init__(self, in_ch=512, out_ch=512, reduce_dim=256, scale=20):
        super(_Up_concat, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.reduce_dim = reduce_dim
        self.scale = scale

        self.reduce256 = nn.Conv2d(self.in_ch, self.reduce_dim, kernel_size=1)
        self.up = nn.ConvTranspose2d(reduce_dim, reduce_dim, 4, stride=2, bias=False)
        # self.Scale_l2_norm = _ScaledL2Norm(2*self.reduce_dim, self.scale)
        self.reduce512 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1_red = F.relu(self.reduce256(x1), inplace=True)
        x1_up = F.relu(self.up(x1_red))
        x2_red = F.relu(self.reduce256(x2), inplace=True)

        # input is NCHW
        diffH = x2_red.size()[2] - x1_up.size()[2]
        diffW = x2_red.size()[3] - x1_up.size()[3]

        x1_up_pad = F.pad(x1_up, (diffW // 2, diffW - diffW // 2,
                                  diffH // 2, diffH - diffH // 2))

        # x = self.Scale_l2_norm(x1_up_pad, x2_red)
        x = torch.cat([x1_up_pad, x2_red], dim=1)
        x_out = F.relu(self.reduce512(x), inplace=True)

        return x_out


class _Context_Generator(nn.Module):
    def __init__(self, w_scale, h_scale, x_rel_pos, y_rel_pos):
        super(_Context_Generator, self).__init__()
        self.w_scale = w_scale
        self.h_scale = h_scale
        self.x_rel_pos = x_rel_pos
        self.y_rel_pos = y_rel_pos
        assert 0 <= self.x_rel_pos <= 1
        assert 0 <= self.y_rel_pos <= 1
        assert 0 <= self.h_scale
        assert 0 <= self.w_scale

    def _get_context_rois(self, im_size, im_rois):
        context_rois = torch.zeros_like(im_rois)
        center_x = (im_rois[:, 0] + im_rois[:, 2]) / 2
        center_y = (im_rois[:, 1] + im_rois[:, 3]) / 2
        half_width = (center_x - im_rois[:, 0]) * self.w_scale
        half_height = (center_y - im_rois[:, 1]) * self.h_scale
        context_rois[:, 0] = torch.from_numpy(np.max(np.hstack(
            (np.zeros([center_x.shape[0], 1]), (center_x - half_width * self.x_rel_pos * 2)[:, np.newaxis])
        ), axis=1))  # x1
        context_rois[:, 1] = torch.from_numpy(np.max(np.hstack(
            (np.zeros([center_x.shape[0], 1]), (center_y - half_height * self.y_rel_pos * 2)[:, np.newaxis])
        ), axis=1))  # y1
        context_rois[:, 2] = torch.from_numpy(np.min(np.hstack(
            (im_size[1] * np.ones([center_x.shape[0], 1]), (context_rois[:, 0] + half_width * 2)[:, np.newaxis])
        ), axis=1))  # x2
        context_rois[:, 3] = torch.from_numpy(np.min(np.hstack(
            (im_size[0] * np.ones([center_x.shape[0], 1]), (context_rois[:, 1] + half_height * 2)[:, np.newaxis])
        ), axis=1))  # y2
        return context_rois

    def forward(self, im_info, im_rois):
        im_size = im_info[0][:2].to('cpu').numpy()  # h, w
        context_rois = self._get_context_rois(im_size, im_rois[0][:, 1:])
        indexs = torch.zeros(context_rois.shape[0], 1)
        indexs = indexs.cuda()
        c_rois = torch.cat([indexs, context_rois], dim=1).unsqueeze(0)
        return c_rois
