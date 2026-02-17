import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSwish(nn.Module):
    def __init__(self, init_beta=1.0):   #like silu but diff equation with a beta trainable
        super().__init__()

        self.beta = nn.Parameter(torch.tensor([float(init_beta)]))

    def forward(self, x):

        return x * torch.sigmoid(self.beta * x)


class Residual_Block(nn.Module):
  def __init__(self ,in_channels:int ,out_channels:int ,dropout=0.1):
    super().__init__()

    self.residual=nn.Sequential(
        nn.GroupNorm(1,in_channels),
        LearnableSwish(),
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.Dropout(dropout),
        nn.GroupNorm(1,out_channels),
        LearnableSwish(),
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),

      )
    if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
            self.shortcut = nn.Identity()

  def forward(self, x):
        return self.shortcut(x) + self.residual(x)




class Upsample(nn.Module):
  def __init__(self,channels):
   super().__init__()
   self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

  def forward(self, x):
   interpolation=F.interpolate(x, scale_factor=2, mode="nearest")

   return self.conv(interpolation)





class Downsample(nn.Module):
  def __init__(self,channels):
   super().__init__()
   self.conv = nn.Conv2d(channels, channels, 3, 2, 0) #stride dec by half instead of max pooling
  def forward(self, x):

    pad=(0,1,0,1)  #Assymetric padding ?

    padded=F.pad(x,pad, mode="constant" ,value=0)

    return self.conv(padded)


class Self_Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.GroupNorm = nn.GroupNorm(1, channels)

        self.query = nn.Conv2d(channels, channels, 1,1,0)   #x*wq   Question

        self.key = nn.Conv2d(channels, channels ,1,1,0) #x*wk looking for

        self.value = nn.Conv2d(channels, channels, 1,1,0) #x*wv what i found

        self.proj = nn.Conv2d(channels, channels, 1,1,0) # feedforward

    def  forward (self,x):
      input=self.GroupNorm(x)
      b, c, h, w = input.shape

      query=self.query(input)

      key=self.key(input)

      value=self.value(input)


      q=query.reshape(b,c,h*w).permute(0,2,1)    #  ((q transpose) *k)
      k=key.reshape(b,c,h*w)
      v=value.reshape(b,c,h*w)

      attn = torch.bmm(q, k)    #Qt *k
      attn = attn * (c**-0.5)
      attn = torch.softmax(attn, dim=-1) # softmax (squareroot Qt * k )

      out = torch.bmm(v, attn.permute(0, 2, 1))   #v*myscores
      out = out.reshape(b, c, h, w)

      out = self.proj(out)

      return x+out
