import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqgan import VQGAN
from models.utils import load_data, weights_init, transform_batch

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator().to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        self.train(args)

    def configure_optimizers(self, args):
        opt_vq = torch.optim.Adam(self.vqgan.parameters(), lr=args.learning_rate, betas=(args.beta, args.beta2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta, args.beta2))
        return opt_vq, opt_disc

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(train_dataset) as pbar:
                for i, batch in enumerate(pbar):
                    imgs = batch[0].to(args.device) if isinstance(batch, (list, tuple)) else batch.to(args.device)
                    imgs = transform_batch(imgs, args.image_size)

                    decoded_images, _, q_loss = self.vqgan(imgs)

                    # Optimize Discriminator
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images.detach())
                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)
                    
                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()
                    self.opt_disc.step()

                    # Optimize VQGAN
                    disc_fake_for_g = self.discriminator(decoded_images)
                    perceptual_rec_loss = (self.perceptual_loss(imgs, decoded_images) + torch.abs(imgs - decoded_images)).mean()
                    g_loss = -torch.mean(disc_fake_for_g)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    self.opt_vq.zero_grad()
                    vq_loss.backward()
                    self.opt_vq.step()

                    if i % 100 == 0:
                        vutils.save_image((torch.cat((imgs[:4], decoded_images[:4])) + 1) / 2, 
                                          f"results/{epoch}_{i}.jpg", nrow=4)

                    pbar.set_postfix(VQ_L=vq_loss.item(), GAN_L=gan_loss.item())
            
            torch.save(self.vqgan.state_dict(), f"checkpoints/vqgan_{epoch}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-codebook-vectors', type=int, default=1024)
    parser.add_argument('--dataset-path', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=2) # Kept low to avoid OOM
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2.25e-05)
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar')
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc-start', type=int, default=10000)
    parser.add_argument('--disc-factor', type=float, default=1.)
    parser.add_argument('--rec-loss-factor', type=float, default=1.)
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.)

    args = parser.parse_args()
    TrainVQGAN(args)
