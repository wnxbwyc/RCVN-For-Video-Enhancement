import torch.nn as nn
import torch

def make_model(opt):
    return rcan_a(opt)

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)

class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            #CoordAtt(num_features, reduction)
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)

class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        num_features = args.n_feats
        num_rg = args.n_resgroups
        num_rcab = args.n_resblocks
        reduction = args.n_feats // 4

        self.rg = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.recons = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.rg(x)
        x = self.recons(x)
        x += residual
        return x


class rcan_a(nn.Module):
    def __init__(self, opt, in_nc=3, out_nc=3, nf = 64):
        super(rcan_a,self).__init__()

        #first
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        #recon(RCAN)
        self.recon_trunk = RCAN(opt)
        #last
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        ref = x
        # recon
        x = self.conv_first(x)
        x = self.recon_trunk(x)
        x = self.conv_last(x)
        x = torch.add(x, ref)
        return x
