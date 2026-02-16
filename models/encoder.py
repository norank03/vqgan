import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper import Residual_Block, Self_Attention, LearnableSwish, Downsample

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        out_channels = args.latent_dim
        
        # Initial Conv: 256x256
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)

        # Level 1: 256x256 -> 128x128
        self.res1 = Residual_Block(128, 128)
        self.res2 = Residual_Block(128, 128)
        self.down1 = Downsample(128) 

        # Level 2: 128x128 -> 64x64
        self.res3 = Residual_Block(128, 256)
        self.res4 = Residual_Block(256, 256)
        self.down2 = Downsample(256) 

        # Level 3: 64x64 -> 32x32
        self.res5 = Residual_Block(256, 512)
        self.down3 = Downsample(512)

        # Level 4: 32x32 -> 16x16
        # ONLY applying attention here at the lowest resolution
        self.res6 = Residual_Block(512, 512)
        self.attn1 = Self_Attention(512)
        self.res7 = Residual_Block(512, 512)
        self.attn2 = Self_Attention(512)
        self.down4 = Downsample(512) # Final resolution 16x16 (if input 256)

        # Final Output Layers
        self.norm_out = nn.GroupNorm(32, 512)
        self.swish = LearnableSwish()
        self.conv_out_last = nn.Conv2d(512, out_channels, 3, 1, 1)

    def forward(self, x):
        # 256x256
        x = self.conv1(x)
        x = self.down1(self.res2(self.res1(x)))
        
        # 128x128
        x = self.down2(self.res4(self.res3(x)))
        
        # 64x64
        x = self.down3(self.res5(x))
        
        # 32x32 -> 16x16
        x = self.attn1(self.res6(x))
        x = self.attn2(self.res7(x))
        x = self.down4(x)

        # Final Processing
        x = self.swish(self.norm_out(x))
        x = self.conv_out_last(x)

        return x
