import torch
import torch.nn as nn
import numpy as np
from fuse import _Context_Generator


im_rois = np.array([[586.7896,  142.5052,  679.2285,  189.9228],
                    [657.7493,  179.4773,  704.8747,  220.5867],
                    [0.0000,  136.3977,  260.1703,  375.0000]])

im_size = np.array([[376.0000, 1245.0000, 1.0027]])

rois = torch.from_numpy(im_rois)
size = torch.from_numpy(im_size)

context_generator = _Context_Generator(2, 2, 0.5, 0.5)
context_generator.forward(im_size, rois)