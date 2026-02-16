import torch
import torch.nn as nn
from models.helper import Residual_Block, Self_Attention, LearnableSwish, Upsample

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        latent_dim = args.latent_dim 
        
        # Initial projection from latent space: 16x16
        self.conv1 = nn.Conv2d(latent_dim, 512, 3, 1, 1)

        # Bottleneck (Attention happens here at 16x16 - very memory efficient)
        self.res1 = Residual_Block(512, 512)
        self.attn1 = Self_Attention(512)
        self.res2 = Residual_Block(512, 512)

        # Upsampling block 1: 16x16 -> 32x32
        self.res3 = Residual_Block(512, 512)
        self.up1 = Upsample(512)

        # Upsampling block 2: 32x32 -> 64x64
        self.res4 = Residual_Block(512, 256)
        self.up2 = Upsample(256)

        # Upsampling block 3: 64x64 -> 128x128
        self.res5 = Residual_Block(256, 128)
        self.up3 = Upsample(128)

        # Upsampling block 4: 128x128 -> 256x256
        self.res6 = Residual_Block(128, 128)
        self.up4 = Upsample(128)

        # Final output
        self.norm = nn.GroupNorm(32, 128)
        self.swish = LearnableSwish()
        self.conv_out = nn.Conv2d(128, 3, 3, 1, 1)

    def forward(self, x):
        # Bottleneck
        x = self.conv1(x)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        
        # Upsampling steps
        x = self.up1(self.res3(x))
        x = self.up2(self.res4(x))
        x = self.up3(self.res5(x))
        x = self.up4(self.res6(x))
        
        # Final reconstruction
        x = self.norm(x)
        x = self.swish(x)
        x = self.conv_out(x)
        return x
