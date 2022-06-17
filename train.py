# implement training loop 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch_optimizer import DiffGrad
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN
from loss import compute_gradient_penalty, pl_reg, r1_reg, wgan_gp_disc_loss, wgan_gp_gen_loss
from utils import random_interpolate_hists
import os

import wandb

wandb.init(project="histogan", entity="metu-kalfa")

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
real_image_dir = "images/anime_face"
# image resolution to generate
image_res = 64
transform = transforms.Compose(
        [transforms.Resize((image_res,image_res)),
        transforms.CenterCrop((image_res,image_res)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# due to hardware limitations similar to paper's authors we kept the batch size small
batch_size = 2
# the dataset contains 63,632 datum, and the we could not make the network to generate meaningfull images therefore kept the epochs small and experimented
num_epochs = 10
# after how many gradient accumulation to optimize parameters
acc_gradient_total = 32
acc_gradient_iter = acc_gradient_total //batch_size
# scalar of R1 regularization
r1_factor = 10
r1_update_iter = 4
# variables for Path length regularization
# please see StyleGAN2 paper B. Implementation Details Path length regularization
ema_decay_coeff = 0.99
target_scale = torch.tensor([0], requires_grad=False).to(device)
plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
plr_update_iter = 8
# after how many iterations to save the nework parameters and generated images
save_iter = 20
# path to save generated images
fake_image_dir = "new_generated_images"
if not os.path.isdir(fake_image_dir):
    os.mkdir(fake_image_dir)
# number of residual blocks in the discriminator 
num_res_blocks = 5
# network capacity to decide the intermediate channel sizes of discrimimator and learnable constant channel size of generator 
network_capacity = 16 
# histogram's bin size
bin_size = 64
# the number of channels are decides as log2(image_res) -1 since we generate 256 res images, there are 7 channels
generator_channel_sizes = [512, 256, 128, 64, 32] # for 64 res images 5 channels
learning_rate = 2e-4
# number of layers for mapping z to w and hist to w_hist
mapping_layer_num = 5
# Initialize Dataset, Dataloader, Discriminator and Generator
dataset = AnimeFacesDataset(real_image_dir, transform, device)
dataloader = DataLoader(dataset, batch_size=acc_gradient_total, shuffle=True, drop_last=True)
generator = HistoGAN(network_capacity, bin_size, mapping_layer_num, generator_channel_sizes)
discriminator = Discriminator(num_res_blocks, network_capacity, image_res)
# discriminator = Discriminator(generator_channel_sizes[::-1], image_res)

# If a pretrained network exists, load their parameters to continue training
# if os.path.exists("generator_new.pt"):
#     generator.load_state_dict(torch.load("generator_new.pt"))
# if os.path.exists("discriminator_new.pt"):
#     discriminator.load_state_dict(torch.load("discriminator_new.pt"))

wandb.watch(discriminator, log_freq=100)
wandb.watch(generator, log_freq=100)
log_interval = 10
log_interval_image = 100

discriminator = discriminator.to(device)
generator=generator.to(device)

# Initialize optimizers 
gene_optim = DiffGrad(generator.parameters(), lr=learning_rate)
disc_optim = DiffGrad(discriminator.parameters(), lr=learning_rate)


def truncation_trick(generator, latent_size, batch_size): # target_hist
    z = torch.randn((1000, latent_size)).to(device)
    w = generator.get_w_from_z(z)
    w_mean = torch.mean(w, dim=0, keepdim=True)
    fake_imgs = generator.gen_image_from_w(w_mean.repeat(batch_size,1), None) # target_hist.size(0), target_hist
    return fake_imgs

def train_generator(generator, discriminator, gene_optim, batch_size, iter): # hist_list
    global target_scale
    total_gene_loss = 0
    total_plr_loss = 0
    gene_optim.zero_grad()
    # for target_hist in hist_list:
    for _ in range(acc_gradient_iter):
        z = torch.randn(batch_size, 512).to(device) # target_hist.size(0)
        fake_data, w = generator(z, None) # target_hist
        disc_score = discriminator(fake_data)
        g_loss = torch.mean(torch.nn.functional.softplus(-disc_score)) / acc_gradient_iter
        # g_loss = - torch.mean(disc_score) / acc_gradient_iter
        g_loss.backward()
        total_gene_loss += g_loss.item()
   
    # if (iter+1) % plr_update_iter == 0:
    #     plr, target_scale = pl_reg(generator, None, target_scale, plr_factor, ema_decay_coeff) # target_hist[0].unsqueeze(0) 
    #     plr.backward() 
    #     total_plr_loss = plr.item()

    # print("Gen loss:", total_gene_loss, "PL loss:", total_plr_loss)
    if iter % log_interval == 0:
        wandb.log({"gen_loss" : total_gene_loss})
    gene_optim.step()
    gene_optim.zero_grad()
    del g_loss, total_gene_loss, total_plr_loss
    # print("gene updated")

def train_discriminator(generator, discriminator, disc_optim, chunk_data, batch_size, iter):
    #hist_list = []
    disc_optim.zero_grad()
    total_real_loss = 0
    total_fake_loss = 0
    # total_disc_loss = 0
    total_r1_loss = 0
    for index in range(chunk_data.size(0)//batch_size):
        batch_data = chunk_data[index*batch_size:(index+1)*batch_size]
        # batch_data = batch_data.to(device)
        #target_hist = random_interpolate_hists(batch_data)
        #hist_list.append(target_hist.clone())
        z = torch.randn(batch_size, 512).to(device)
        fake_data, _ = generator(z, None) #target_hist
        fake_data = fake_data.detach()
        fake_scores = discriminator(fake_data)
        real_scores = discriminator(batch_data)
        # gradient_penalty = compute_gradient_penalty(fake_data, batch_data, discriminator)
        real_loss = torch.mean(torch.nn.functional.softplus(-real_scores))/ acc_gradient_iter  # -torch.mean(real_scores) / acc_gradient_iter #  
        fake_loss = torch.mean(torch.nn.functional.softplus(fake_scores)) / acc_gradient_iter # torch.mean(fake_scores) /  acc_gradient_iter  #  
        disc_loss = real_loss + fake_loss  #torch.mean(torch.nn.functional.relu(1 + real_scores) + torch.nn.functional.relu(1 - fake_scores)) 
        disc_loss.backward()
        # total_disc_loss += disc_loss.item()
        total_fake_loss += fake_loss.item()
        total_real_loss += real_loss.item()

    # if (iter+1) % r1_update_iter == 0:
        # r1_loss =  r1_reg(batch_data, discriminator, r1_factor)
        # r1_loss.backward()
        # total_r1_loss = r1_loss.item()
        # wandb.log({"r1_loss" : total_r1_loss})


    # print("Real loss:", total_real_loss-total_r1_loss, "Fake loss:", total_fake_loss, "R1 loss:", total_r1_loss) 
    # print("Disc Loss:", total_disc_loss, "R1 loss:", total_r1_loss) 
    if iter % log_interval == 0:
        wandb.log({"disc_fake_loss" : total_fake_loss, "disc_real_loss": total_real_loss})

    disc_optim.step()
    disc_optim.zero_grad()
    del disc_loss, total_fake_loss, total_real_loss, total_r1_loss
    # print("disc updated")
    #return hist_list
        


# Traning loop without gradient accumulation
# Gradient accumulation is implemented and tried in train2.py but had some performance and memory consumption issues therefore not added here
dataset_size = len(dataset)
for epoch in range(0, num_epochs):
    for iter, chunk_data in enumerate(dataloader):
        torch.cuda.empty_cache()
        # print(batch_data.max(), batch_data.min())
        training_percent = 100*iter*chunk_data.size(0)/dataset_size
        wandb.log({"training_percent": training_percent})
        # print("Epoch",epoch, " Training %", training_percent)
        train_discriminator(generator, discriminator, disc_optim, chunk_data, batch_size, iter) # hist_list = 
        train_generator(generator, discriminator, gene_optim, batch_size, iter)

        if iter % log_interval_image:
            fake_data = torch.clamp(truncation_trick(generator, 512, batch_size), 0, 1) # hist_list[-1]
            # z = torch.randn(batch_size, 512).to(device)
            # fake_data = generator(z, None)
            wandb.log({"generated_image": wandb.Image(fake_data[0])})            
        if (iter+1) % save_iter == 0:
            # for i in range(fake_data.size(0)):
            #     save_image(fake_data[i], os.path.join(fake_image_dir, "fake_{}_{}_{}.png".format(epoch, iter, i)))
            torch.save(generator.state_dict(), "generator_new.pt")
            torch.save(discriminator.state_dict(), "discriminator_new.pt")
        





