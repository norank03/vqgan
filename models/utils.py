import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class ImagePaths(Dataset):
    def __init__(self, path):
        self.images = [os.path.join(path, file) for file in os.listdir(path)
                       if file.endswith(('.png', '.jpg', '.jpeg'))]
        self._length = len(self.images)

        self.resizer = transforms.ToTensor()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert("RGB")
        return self.resizer(image)

def load_data(args):
  dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                   transform=transforms.ToTensor())
  train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


  return train_loader

def transform_batch(batch, size):

    if isinstance(batch, (list, tuple)):
        batch = batch[0]

    # 2. Define GPU-compatible transforms
    # Note: Using 'bicubic' interpolation is usually better for GANs
    resizer = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)
    normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    batch = resizer(batch)
    batch = normalizer(batch)
    return batch





def weights_init(m):   #LIKE xavier
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)




def plot_images(images_dict):
    """
    Corrected plotting: Un-normalizes images for display.
    """
    keys = ["input", "rec", "half_sample", "full_sample"]
    # Filter keys that actually exist in the dict
    keys = [k for k in keys if k in images_dict]

    fig, axarr = plt.subplots(1, len(keys), figsize=(15, 5))

    for i, key in enumerate(keys):
        # Take first image in batch, move to CPU, HWC format
        img = images_dict[key][0].cpu().detach().numpy().transpose(1, 2, 0)
        # IMPORTANT: Scale from [-1, 1] back to [0, 1] for matplotlib
        img = np.clip((img + 1.0) / 2.0, 0, 1)

        if len(keys) > 1:
            axarr[i].imshow(img)
            axarr[i].set_title(key)
            axarr[i].axis("off")
        else:
            axarr.imshow(img)
            axarr.axis("off")

    plt.show()
