# implement training loop 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch_optimizer import DiffGrad
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN
from loss import compute_gradient_penalty, pl_reg, gp_only_real, wgan_gp_disc_loss, wgan_gp_gen_loss, hellinger_dist_loss
from utils import random_interpolate_hists
import os

import wandb
config = dict(
    num_epochs = 20,
    batch_size = 16,
    acc_gradient_total = 16,
    r1_factor = 10,
    r1_update_iter = 4,
    decay_coeff = 0.99,
    plr_update_iter = 32,
    save_iter = 400,
    image_res = 64,
    network_capacity = 16,
    latent_dim = 512,
    bin_size = 64,
    learning_rate = 0.0002,
    mapping_layer_num = 8,
    mixing_prob = 0.9,
    use_plr = True,
    use_r1r = True,
    kaiming_init=True,
    use_eqlr = False,
    use_spec_norm = False,
    disc_arch= "ResBlock",
    gen_arch = "InputModDemod",
    optim="Adam",
    loss_type="wasser")

# wandb.init(project="histogan", 
#            entity="metu-kalfa",
#            config = config)

# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.benchmark = True

# parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
real_image_dir = "images"
# image resolution to generate
image_res = config["image_res"]
transform = transforms.Compose(
        [transforms.Resize((image_res,image_res))])

# due to hardware limitations similar to paper's authors we kept the batch size small
batch_size = config["batch_size"]
# the dataset contains 63,632 datum, and the we could not make the network to generate meaningfull images therefore kept the epochs small and experimented
num_epochs = config["num_epochs"]
# after how many gradient accumulation to optimize parameters
acc_gradient_total = config["acc_gradient_total"]
acc_gradient_iter = acc_gradient_total // batch_size
# scalar of R1 regularization
r1_factor = config["r1_factor"]
r1_update_iter = config["r1_update_iter"]
# variables for Path length regularization
# please see StyleGAN2 paper B. Implementation Details Path length regularization
decay_coeff = config["decay_coeff"]
target_scale = torch.tensor([0], requires_grad=False).to(device)
plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
plr_update_iter = config["plr_update_iter"]
# after how many iterations to save the nework parameters and generated images
save_iter = config["save_iter"]
# path to save generated images
fake_image_dir = "new_generated_images"
if not os.path.isdir(fake_image_dir):
    os.mkdir(fake_image_dir)
# network capacity to decide the intermediate channel sizes of discrimimator and learnable constant channel size of generator 
network_capacity = config["network_capacity"] 
# noise latent dim size
latent_dim = config["latent_dim"]
# histogram's bin size
bin_size = config["bin_size"]
learning_rate = config["learning_rate"]
# number of layers for mapping z to w and hist to w_hist
mapping_layer_num = config["mapping_layer_num"]
# probability of using mixed noise
mixing_prob = config["mixing_prob"]
num_gen_layers = int(np.log2(image_res)-1)
use_plr = config["use_plr"]
use_r1r = config["use_r1r"]
kaiming_init= config["use_r1r"]
use_eqlr = config["use_eqlr"]
use_spec_norm = config["use_spec_norm"]
loss_type = config["loss_type"]
optim = config["optim"]
# Initialize Dataset, Dataloader, Discriminator and Generator
dataset = AnimeFacesDataset(real_image_dir, transform, device)
dataloader = DataLoader(dataset, batch_size=acc_gradient_total, shuffle=True, drop_last=True)
generator = HistoGAN(network_capacity, latent_dim, bin_size, image_res, mapping_layer_num, kaiming_init=kaiming_init, use_eqlr=use_eqlr)
discriminator = Discriminator(network_capacity, image_res, kaiming_init=kaiming_init, use_spec_norm=use_spec_norm)

import sys
with open('generator.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(generator)
with open('discriminator.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(discriminator)
sys.stdout = sys.__stdout__

# If a pretrained network exists, load their parameters to continue training
# if os.path.exists("generator_new.pt"):
#     generator.load_state_dict(torch.load("generator_new.pt"))
# if os.path.exists("discriminator_new.pt"):
#     discriminator.load_state_dict(torch.load("discriminator_new.pt"))


log_interval = 200

discriminator = discriminator.to(device)
generator=generator.to(device)

# Initialize optimizers 
if optim == "Adam":
    gene_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.99))
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.99))
elif optim == "DiffGrad":
    gene_optim = DiffGrad(generator.parameters(), lr=learning_rate)
    disc_optim = DiffGrad(discriminator.parameters(), lr=learning_rate)


def truncation_trick(generator, latent_size, batch_size): # target_hist
    with torch.no_grad():
        z = torch.randn((1000, latent_size)).to(device)
        w = generator.get_w_from_z(z)
        w_mean = torch.mean(w, dim=0, keepdim=True)
        fake_imgs = generator.gen_image_from_w(w_mean.repeat(batch_size,1), None) # target_hist.size(0), target_hist
    return fake_imgs

def mixing_noise():
    if torch.rand((1,)) < mixing_prob:
        ri = torch.randint(1, num_gen_layers, (1,)).item()
        z = torch.cat([torch.randn((batch_size, 1, latent_dim)).expand(-1,ri,-1), torch.randn((batch_size, 1, latent_dim)).expand(-1,num_gen_layers-ri,-1)], dim=1)
    else:
        z = torch.randn((batch_size, num_gen_layers, latent_dim))
    return z


def train_generator(generator, discriminator, gene_optim, batch_size, iter, hist_list):
    global target_scale
    total_gene_loss = 0
    total_plr_loss = 0
    gene_optim.zero_grad()
    for target_hist in hist_list:
        for _ in range(acc_gradient_iter):
            z = mixing_noise().to(device) # torch.randn(batch_size, latent_dim)
            fake_data, w = generator(z, target_hist) # target_hist
            disc_score = discriminator(fake_data)
            if loss_type in ["wasser", "hinge"]:
                g_loss = -torch.mean(disc_score) / acc_gradient_iter
            elif loss_type == "softplus":
                g_loss = torch.mean(torch.nn.functional.softplus(-disc_score)) / acc_gradient_iter      
            total_gene_loss += g_loss.item()
            pl_loss = 0 
            if use_plr and (iter+1) % plr_update_iter == 0:
                std = 0.1 / (w.std(dim=0, keepdim=True) + 1e-8)
                w_changed = w + torch.randn_like(w, device=device) / (std + 1e-8)
                changed_data = generator.gen_image_from_w(w_changed, target_hist)
                pl_lengths = ((changed_data - fake_data) ** 2).mean(dim=(1, 2, 3))
                avg_pl_length = torch.mean(pl_lengths).item()
                pl_loss = torch.mean(torch.square(pl_lengths - target_scale))
                # plr, target_scale = pl_reg(generator, None, target_scale, plr_factor, decay_coeff) # target_hist[0].unsqueeze(0) 
                total_plr_loss += pl_loss.item()

            g_loss += pl_loss
            g_loss.backward()
        gene_optim.step()
        gene_optim.zero_grad()
    # if use_plr and (iter+1) % plr_update_iter == 0:
    #     target_scale = (1-decay_coeff)* target_scale + decay_coeff * avg_pl_length
    #     wandb.log({"pl_loss": total_plr_loss}, step=iter)


    # if iter % log_interval == 0:
    #     wandb.log({"gen_loss" : total_gene_loss}, step=iter)
    
    del g_loss, total_gene_loss, total_plr_loss

def train_discriminator(generator, discriminator, disc_optim, chunk_data, batch_size, iter):
    hist_list = []
    disc_optim.zero_grad()
    total_disc_loss = 0
    total_real_loss = 0
    total_fake_loss = 0
    total_r1_loss = 0
    for index in range(chunk_data.size(0)//batch_size):
        batch_data = chunk_data[index*batch_size:(index+1)*batch_size]
        batch_data.requires_grad_()
        batch_data = batch_data.to(device)
        target_hist = random_interpolate_hists(batch_data)
        hist_list.append(target_hist.clone())
        z = mixing_noise().to(device) # torch.randn(batch_size, latent_dim)
        fake_data, _ = generator(z, target_hist) #target_hist
        fake_data = fake_data.detach()
        fake_scores = discriminator(fake_data)
        real_scores = discriminator(batch_data)
        if loss_type == "hinge":
            real_loss = torch.mean(torch.nn.functional.relu(1-real_scores))    
            fake_loss = torch.mean(torch.nn.functional.relu(1+ fake_scores)) 
        elif loss_type == "softplus":
            real_loss = torch.mean(torch.nn.functional.softplus(-real_scores))/ acc_gradient_iter
            fake_loss = torch.mean(torch.nn.functional.softplus(fake_scores)) / acc_gradient_iter
        elif loss_type == "wasser":
            real_loss = -torch.mean(real_scores) / acc_gradient_iter 
            fake_loss = torch.mean(fake_scores) /  acc_gradient_iter


        disc_loss =  real_loss + fake_loss  # torch.mean(torch.nn.functional.relu(1 + real_scores) + torch.nn.functional.relu(1 - fake_scores)) #
        total_disc_loss += disc_loss.item()
        total_fake_loss += fake_loss.item()
        total_real_loss += real_loss.item()
        r1_loss = 0
        if use_r1r and iter % r1_update_iter == 0:
            r1_loss =  gp_only_real(batch_data, real_scores, r1_factor)/ acc_gradient_iter
            total_r1_loss += r1_loss.item()

        real_loss += r1_loss        
        real_loss.backward()
        fake_loss.backward()

    # if iter % log_interval == 0:
    #     wandb.log({"disc_loss" : total_disc_loss},step=iter)
    #     if use_r1r and iter % r1_update_iter == 0:
    #        wandb.log({"r1_loss" : total_r1_loss},step=iter)


    disc_optim.step()
    disc_optim.zero_grad()
    del disc_loss, total_disc_loss, total_r1_loss
    return hist_list


# # Traning loop without gradient accumulation
# # Gradient accumulation is implemented and tried in train2.py but had some performance and memory consumption issues therefore not added here
dataset_size = len(dataset)
total_iter = 0
for epoch in range(0, num_epochs):
    for iter, chunk_data in enumerate(dataloader):
        # print(batch_data.max(), batch_data.min())
        training_percent = 100*iter*chunk_data.size(0)/dataset_size
        
        # print("Epoch",epoch, " Training %", training_percent)
        total_iter += 1
        hist_list = train_discriminator(generator, discriminator, disc_optim, chunk_data, batch_size, total_iter) # 
        train_generator(generator, discriminator, gene_optim, batch_size, total_iter, hist_list)

        if iter % log_interval == 0:
            # wandb.log({"training_percent": training_percent},step=iter)
            # torch.clamp(truncation_trick(generator, latent_dim, batch_size), 0, 1) # hist_list[-1]
            z = mixing_noise().to(device)
            fake_data, _ = generator(z, hist_list[-1])
            # print(type(fake_data), fake_data)
            save_image(fake_data, os.path.join(fake_image_dir, "fake_{}_{}_norm.png".format(epoch, iter)), normalize=True)
            # save_image(chunk_data, os.path.join(fake_image_dir, "real_{}_{}_norm.png".format(epoch, iter)), normalize=True)

            # save_image(torch.clamp(fake_data, 0, 1), os.path.join(fake_image_dir, "fake_{}_{}_clamp.png".format(epoch, iter)))

            print(training_percent)
            #wandb.log({"generated_image": wandb.Image(fake_data[0])}, step=iter)            
        if (iter+1) % save_iter == 0:
            torch.save(generator.state_dict(), "generator_new.pt")
            torch.save(discriminator.state_dict(), "discriminator_new.pt")
        