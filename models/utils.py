import os
import albumentations
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size
        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        # Scale to [-1, 1] for VQGAN
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        return self.preprocess_image(self.images[i])

class CIFAR10Dataset(Dataset):
    """Wrapper to make CIFAR-10 behave like ImagePaths (returns just the image)"""
    def __init__(self, train=True, size=256):
        # We use torchvision's downloader, then apply your VQGAN scaling
        self.data = datasets.CIFAR10(root='./data', train=train, download=True)
        self.size = size
        
        self.rescaler = albumentations.Resize(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, _ = self.data[i] # Drop the label
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image.transpose(2, 0, 1)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_data(args):
    """
    args.dataset_path should be 'cifar10' to use the benchmark, 
    otherwise it should be a path to a folder of images.
    """
    if args.dataset_path.lower() == "cifar10":
        train_data = CIFAR10Dataset(train=True, size=args.image_size)
    else:
        train_data = ImagePaths(args.dataset_path, size=args.image_size)
        
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return train_loader

# -----------------------------------------------------------------------------
# Model Utils
# -----------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_images(images_dict):
    """
    Plots: Input, Reconstruction, Half Sample, Full Sample.
    Expects tensors in range [-1, 1].
    """
    keys = ["input", "rec", "half_sample", "full_sample"]
    fig, axarr = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, key in enumerate(keys):
        if key in images_dict:
            # Take first image in batch, move to CPU, transpose to (H, W, C)
            img = images_dict[key][0].cpu().detach().numpy().transpose(1, 2, 0)
            # Re-scale from [-1, 1] to [0, 1] for matplotlib
            img = np.clip((img + 1.0) / 2.0, 0, 1)
            axarr[i].imshow(img)
            axarr[i].set_title(key)
            axarr[i].axis("off")
            
    plt.tight_layout()
    plt.show()
