import torch 
import torch.nn as nn
import torch.nn.functional as F
from helper import Residual_Block, Self_Attention, LearnableSwish, Upsample, Downsample

class Decoder(nn.Module):
  def __init__(self,latent_dim=256,img_channels=3):
    super().__init__()

    self.conv1=nn.Conv2d(latent_dim,256,3,1,1)

    self.res1=Residual_Block(256,256)
    self.attn1=Self_Attention(256)
    self.res2=Residual_Block(256,256)


    self.res3=Residual_Block(256,256)
    self.res4=Residual_Block(256,256)
    self.up1=self.Upsample(256)

    self.res5=Residual_Block(256,128)
    self.attn2=Self_Attention(128)
    self.res6=Residual_Block(128,128)
    self.up2=self.Upsample(128)

    self.res7=Residual_Block(128,128)
    self.res8=Residual_Block(128,128)

    self.normout=nn.GroupNorm(32,128)
    self.swish=LearnableSwish()

    self.conv_out=nn.Conv2d(128,img_channels,3,1,1)

    def forward(self,x):
     x=self.conv1(x)
     x=self.res1(x)
     x=self.attn1(x)
     x=self.res2(x)

     
     x=self.res3(x)
     x=self.res4(x)
     x=self.up1(x)

     x=self.res5(x)
     x=self.attn2(x)
     x=self.res6(x)
     x=self.up2(x)

     x=self.res7(x)
     x=self.res8(x)
    
     x=self.normout(x)
     x=self.swish(x)
     x=self.conv_out(x)

     return x

    




