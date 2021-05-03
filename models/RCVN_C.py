import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import functools
import torch
from models.RCAN_A import rcan_a
from models.RCVN_B import rcvn


def make_model(opt):
    return rcvn_c(opt)

# rcvn_c
class rcvn_c(nn.Module):
    def __init__(self, opt, in_nc=3, out_nc=3, nf = 64, nb = 16):
        super(rcvn_c,self).__init__()
        num_channels = 3

        self.rcvn_b = rcvn(opt)
        self.rcvn_b.load_state_dict(torch.load("rcvn_b_3_19.pt"))


    def forward(self, x, neilr):
        b, c, h, w = x.size()
        x = self.rcvn_b(x, neilr)
        return x




