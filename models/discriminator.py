import torch
import torch.nn as nn
class Discriminator(nn.Module):   # classify each pixel in he receptive field fake or real bad at whole image high level
    def __init__(self, image_channels=3):
        super().__init__()

        # Layer 1: [3, 32, 32] -> [64, 16, 16]
        # (Stride 2 halves the size)
        self.l1 = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 2: [64, 16, 16] -> [128, 8, 8]
        # (Stride 2 halves the size again)
        self.l2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 3: [128, 8, 8] -> [256, 4, 4]
        # (Stride 2 halves the size again)
        self.l3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Layer 4: [256, 4, 4] -> [512, 3, 3]
        # (Note: Stride 1 here, size changes slightly due to kernel/padding)
        self.l4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final Layer: [512, 3, 3] -> [1, 2, 2]
        # This outputs a small grid of "Real/Fake" scores  Small receptive field ?   # what is the diff 2x2 ?
        self.conv_out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.conv_out(x)
