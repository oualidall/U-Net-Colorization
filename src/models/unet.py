import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Bloc de convolution avec normalisation configurable ----
def conv_block(cin, cout, norm="bn", dropout=0.0):
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.InstanceNorm2d(cout) if norm=="in" else nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    layers += [
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.InstanceNorm2d(cout) if norm=="in" else nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)

# ---- U-Net principal ----
class UNet(nn.Module):
    def __init__(self, in_ch=1, base=64, norm="bn", dropout=0.0):
        super().__init__()
        self.enc1 = conv_block(in_ch, base, norm, dropout)
        self.enc2 = conv_block(base, base*2, norm, dropout)
        self.enc3 = conv_block(base*2, base*4, norm, dropout)
        self.enc4 = conv_block(base*4, base*8, norm, dropout)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*8, base*16, norm, dropout)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8, norm, dropout)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4, norm, dropout)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2, norm, dropout)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base, norm, dropout)

        self.head = nn.Conv2d(base, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
