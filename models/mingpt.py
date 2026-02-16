
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size  # Number of tokens
        self.block_size = block_size # max tokens i can take
        for k, v in kwargs.items():
            setattr(self, k, v)





class CausalSelfAttention(nn.Module):
  def __init__(self,config) :
      super().__init__()

      self.n_embd = config.n_embd
      self.n_head = config.n_head
      self.head_dim = config.n_embd // config.n_head

      self.key = nn.Linear(self.n_embd, self.n_embd)
      self.query = nn.Linear(self.n_embd, self.n_embd)
      self.value = nn.Linear(self.n_embd, self.n_embd)


      self.proj = nn.Linear(self.n_embd, self.n_embd)

      self.attn_drop = nn.Dropout(config.attn_pdrop)
      self.resid_drop = nn.Dropout(config.resid_pdrop)

      self.proj = nn.Linear(config.n_embd, config.n_embd)

      mask = torch.tril(torch.ones(config.block_size, config.block_size))
      self.register_buffer("mask", mask)

  def forward (self):
     B, T, C = x.shape  # batch time chn

     k = self.key(x)
     q = self.query(x)
     v = self.value(x)

     k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
     q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
     v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

     att = q @ k.transpose(-2, -1)
     att = att / math.sqrt(self.head_dim)   #qktrans

     att = att.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )     # infenity to softmax 0



     att = F.softmax(att, dim=-1)

     y = att @ v   #scores the values

     y = y.transpose(1, 2).contiguous()
     y = y.view(B, T, C)

     y = self.proj(y)  #feed forward

     return y




class Block (nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)       #layer norm ---- attn -----mlp (feedforward )
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)

        # Feed Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),   # Activation
            nn.Linear(4 * config.n_embd, config.n_embd),
        )


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd  )

        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)    # for each position hn3ml 256 dim
        )


        self.block1 = Block(config)
        self.block2 = Block(config)


        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size
        )

        self.block_size = config.block_size

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)

        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        x = self.block1(x)
        x = self.block2(x)

        x = self.ln_f(x)


        logits = self.head(x)

        return logits




