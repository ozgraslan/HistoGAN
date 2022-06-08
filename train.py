# implement training loop 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from data import AnimeFacesDataset
from model import Discriminator, HistoGAN
from loss import compute_gradient_penalty, pl_reg, r1_reg, wgan_gp_disc_loss, wgan_gp_gen_loss
from utils import random_interpolate_hists
import os

from torchinfo import summary
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
real_image_dir = "images"
transform = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(0.5)])
dataset = AnimeFacesDataset(real_image_dir, transform, device)
# due to hardware limitations similar to paper's authors we kept the batch size small
batch_size = 2
# the dataset contains 63,632 datum, and the we could not make the network to generate meaningfull images therefore kept the epochs small and experimented
num_epochs = 2
# variable to hold after how many discriminator updates to update the generator
g_update_iter = 5
# after how many gradient accumulation to optimize parameters
acc_gradient_iter = 1
# scalar of R1 regularization
r1_factor = 10
# variables for Path length regularization
# please see StyleGAN2 paper B. Implementation Details Path length regularization
ema_decay_coeff = 0.99  # 0.999 for previous works
target_scale = torch.tensor([0]).to(device)
plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
# after how many iterations to save the nework parameters and generated images
save_iter = 200
# path to save generated images
fake_image_dir = "generated_images"
if not os.path.isdir(fake_image_dir):
    os.mkdir(fake_image_dir)
# number of residual blocks in the discriminator 
num_res_blocks = 7
# network capacity to decide the intermediate channel sizes of discrimimator and learnable constant channel size of generator 
network_capacity = 16 
# histogram's bin size
bin_size = 64
# the number of channels are decides as log2(image_res) -1 since we generate 256 res images, there are 7 channels
generator_channel_sizes = [1024, 512, 512, 512, 256, 128, 64]
learning_rate = 2e-4
# coefficient of gradient penalty
coeff_penalty = 10 # same as the StyleGAN2 paper

# Initialize Dataset, Dataloader, Discriminator and Generator
dataset = AnimeFacesDataset(real_image_dir, transform, device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
generator = HistoGAN(network_capacity, bin_size, generator_channel_sizes)
discriminator = Discriminator(num_res_blocks, network_capacity)

# If a pretrained network exists, load their parameters to continue training
if os.path.exists("generator.pt"):
    generator.load_state_dict(torch.load("generator.pt"))
if os.path.exists("discriminator.pt"):
    discriminator.load_state_dict(torch.load("discriminator.pt"))


discriminator = discriminator.to(device)
generator=generator.to(device)

# Initialize optimizers 
gene_optim = torch.optim.Adam(generator.parameters(), lr= learning_rate)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr= learning_rate)

# Traning loop without gradient accumulation
# Gradient accumulation is implemented and tried in train2.py but had some performance and memory consumption issues therefore not added here
for epoch in range(num_epochs):
    for iter, batch_data in enumerate(dataloader):
        # torch.cuda.empty_cache() 
        training_percent = 100*iter*batch_data.size(0)/len(dataset)
        batch_data = batch_data.to(device)
        # Sample random Gaussian noise
        z = torch.randn(batch_data.size(0), 512).to(device)
        # Interpolate between target image histogram 
        # to prevent overfitting to dataset images
        target_hist = random_interpolate_hists(batch_data)
        # Generate fake images
        fake_data, w = generator(z, target_hist)

        # Detach fake data so no gradient accumalition 
        # to generator while only training discriminator
        fake_data = fake_data.detach()

        # Compute real probabilities computed by discriminator
        fake_scores = discriminator(fake_data)
        real_scores = discriminator(batch_data)
        gradient_penalty = compute_gradient_penalty(fake_data, batch_data, discriminator)
        d_loss = wgan_gp_disc_loss(real_scores, fake_scores, gradient_penalty, coeff_penalty)
        #d_loss = disc_loss(fake_scores, real_scores)
        # in stylegan2 paper they argue applying regularization in every 16 iteration does not hurt perfrormance 
        if (iter+1) % 16 == 0: 
            # r1 regulatization
            d_loss = d_loss + r1_reg(batch_data, discriminator, r1_factor)  

        print("%", training_percent, " Disc loss:", d_loss.item())
        d_loss.backward()
        disc_optim.step()
        disc_optim.zero_grad()

        if (iter+1) % g_update_iter == 0:
            z = torch.randn(batch_data.size(0), 512).to(device)
            fake_data, w = generator(z, target_hist) 

            disc_score = discriminator(fake_data)
            g_loss = wgan_gp_gen_loss(disc_score)
            if (iter+1) % (8*g_update_iter) == 0:
                plr, ema_decay_coeff = pl_reg(fake_data, w, target_scale, plr_factor, ema_decay_coeff)
                g_loss = g_loss + plr

            print("%", training_percent, "Gen loss:", g_loss.item())
            g_loss.backward()
            gene_optim.step()
            gene_optim.zero_grad()
            
        if (iter+1) % save_iter == 0:
            for i in range(fake_data.size(0)):
                save_image(fake_data[i], os.path.join(fake_image_dir, "fake_{}_{}_{}.png".format(epoch, iter, i)))
            torch.save(generator.state_dict(), "generator.pt")
            torch.save(discriminator.state_dict(), "discriminator.pt")
        





