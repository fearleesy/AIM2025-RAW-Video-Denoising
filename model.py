import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )

def up_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )

class UNetDenoise(nn.Module):
    def __init__(self, in_channels=40, base=64, out_channels=4):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
        self.enc2 = conv_block(base, base*2)
        self.enc3 = conv_block(base*2, base*4)

        self.bottleneck = conv_block(base*4, base*8)

        self.up3 = up_conv(base*8, base*4)
        self.dec3 = conv_block(base*8, base*4)

        self.up2 = up_conv(base*4, base*2)
        self.dec2 = conv_block(base*4, base*2)

        self.up1 = up_conv(base*2, base)
        self.dec1 = conv_block(base*2, base)

        self.final = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)