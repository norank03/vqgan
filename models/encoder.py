import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import Residual_Block, Self_Attention, LearnableSwish, Upsample, Downsample


class Encoder(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.conv1=nn.Conv2d(3,128,kernel_size=3,padding=1)

    #block1 32 to 16
    self.res1=Residual_Block(128,128)
    self.res2=Residual_Block(128,128)
    self.Downsample(128)


    #block2 16 to 8
    self.res3=Residual_Block(128,256)
    self.attn1=Self_Attention(256)
    self.res4=Residual_Block(256,256)
    self.Downsample(256)


    #bottelneck Attention
    self.res5=Residual_Block(256,256)
    self.attn2=Self_Attention(256)
    self.res6=Residual_Block(256,256)

    self.norm_out = nn.GroupNorm(32, 256)
    self.swish = LearnableSwish()
    self.conv_out = nn.Conv2d(256, 256, 3, 1, 1)

    self.norm=nn.GroupNorm(32,256)
    self.swish=LearnableSwish()
    self.conv_out_last=nn.Conv2d(256,out_channels,3,1,1)

def forward(self,x):
  x=self.conv1(x)

  x=self.res1(x)
  x=self.res2(x)
  x=self.Downsample(x)

  x=self.res3(x)
  x=self.attn1(x)
  x=self.res4(x)
  x=self.Downsample(x)

  x=self.res5(x)
  x=self.attn2(x)
  x=self.res6(x)

  x=self.norm_out(x)
  x=self.swish(x)
  x=self.conv_out(x)

  x=self.norm(x)
  x=self.swish(x)
  x=self.conv_out_last(x)

  return x






