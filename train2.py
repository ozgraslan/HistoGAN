""" This is the playground code. Modular and commented version of training is train.py
"""

# implement training loop 
from locale import normalize
from statistics import variance
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from torch_optimizer import DiffGrad
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN, HistoGANAda
from loss import wgan_gp_gen_loss, wgan_gp_disc_loss, non_sat_generator_loss, gradient_penalty
from utils import histogram_feature_v2

from functools import partial
import os
import wandb

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
# wandb.init(project="histogan", entity="metu-kalfa")

def cycle(iterable):
  while True:
    for i in iterable:
      yield i

def truncation_trick(generator, latent_size, batch_size, batch_data): # target_hist
    z = torch.randn((1000, latent_size)).to(device)
    w = generator.get_w_from_z(z)
    w_mean = torch.mean(w, dim=0, keepdim=True)
    target_hist = random_interpolate_hists(batch_data)
    fake_imgs = generator.gen_image_from_w(w_mean.repeat(batch_size,1), target_hist) # target_hist.size(0), target_hist
    return fake_imgs

def random_interpolate_hists(batch_data):
    B = batch_data.size(0)
    delta = torch.rand((B,1,1,1)).to(device)
    first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    first_hist = histogram_feature_v2(first_images)
    second_hist = histogram_feature_v2(second_images)
    hist_t = delta*first_hist + (1-delta)*second_hist
    return hist_t

def compute_gradient_penalty(fake_data, real_data, discriminator):
    a = torch.rand((fake_data.size(0), 1, 1, 1)).to(device)
    comb_data = a* fake_data + (1-a)*real_data
    comb_data = comb_data.requires_grad_(True)
    comb_score = discriminator(comb_data)
    gradients = torch.autograd.grad(outputs=comb_score, inputs=comb_data, grad_outputs=torch.ones(comb_score.size()).to(device), create_graph=True, retain_graph=True)[0]
    gradient_norm = torch.sqrt(torch.sum(torch.square(gradients.view(gradients.size(0), -1)), dim=1))
    gradient_penalty = torch.mean(torch.square(gradient_norm-1))
    return gradient_penalty


device = "cuda" if torch.cuda.is_available() else "cpu"
real_image_dir = "images"
transform = transforms.Compose(
    [transforms.Resize((64,64))])
dataset = AnimeFacesDataset(real_image_dir, transform, device)

batch_size = 16
dataloader = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True))
num_epochs = 25

generator = HistoGAN()
discriminator = Discriminator(7, 16)

# generator.load_state_dict(torch.load("models/generator_8.pt"))

# with open("gen_params.txt", "w") as f:
#     for name, param in generator.named_parameters():
#         f.write(name + " " + str(torch.mean(param)) + " " +str(torch.var(param)) + "\n")
#     f.write("\n")
        
# with open("disc_params.txt", "w") as f:
#     for name, param in generator.named_parameters():
#         f.write(name + " " + str(torch.mean(param)) + " " +str(torch.var(param)) + "\n")
#     f.write("\n")

gen_params = []
disc_params = []

for name, param in generator.named_parameters():
    param = param.detach().numpy()
    gen_params.append([np.mean(param), np.var(param)])
for name, param in discriminator.named_parameters():
    param = param.detach().numpy()
    disc_params.append([np.mean(param), np.var(param)])

gen_params = np.array(gen_params)
disc_params = np.array(disc_params)

discriminator = discriminator.to(device)
generator=generator.to(device)

g_update_iter = 1
acc_gradient_iter = 4
r1_factor = 10

# Improvement notes
# We can not get a good generator and followings may be the origin:
# Initialize with He explicitly, we guess Pytorch does in this way 
#   but in some deep learning projects they forced to initialize with He.
# EMA initialization starts with 0 but starting it with respect to initial
#   wieghts may help
# Paper does not mention learnign rate decay, unlike Stylegan 2 uses a decay policy

# Further improvement
# Path length penalty for better latent space
#   will be added after reaching visually explainable results

# please see StyleGAN2 paper B. Implementation Details Path length regularization
ema_decay_coeff = 0.99
target_scale = None # torch.tensor([0]).to(device)  # Should not be 0, first weights should be used
plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
coeff_penalty = 10 # same as the StyleGAN2 paper

save_iter = 100

# gene_optim = torch.optim.Adam(
#     generator.parameters(),
#     lr= 2e-4,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=0,
# )
# disc_optim = torch.optim.Adam(
#     discriminator.parameters(),
#     lr= 2e-4,
#     betas=(0.9, 0.999),
#     eps=1e-8,
#     weight_decay=0,
# )

gene_optim = DiffGrad(
    generator.parameters(),
    lr= 2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)
disc_optim = DiffGrad(
    discriminator.parameters(),
    lr= 2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

fake_image_dir = "generated_images"
if not os.path.isdir(fake_image_dir):
    os.mkdir(fake_image_dir)

seen = 0
disc_step = 0
update_disc = 8
print("Starting to train")
break_flag = False
for epoch in range(num_epochs):
    seen = 0
    while seen < len(dataset):
        torch.cuda.empty_cache()
        cumulative_dloss = 0.0
        cum_gp = 0.0
        disc_optim.zero_grad()
        if break_flag: 
            break_flag=False
            break
        # for _ in range(update_disc): 
        training_percent = 100*seen/len(dataset)
        batch_data = next(dataloader)
        if batch_data.size(0)==1:
            seen += 1
            break_flag = True
            break
        seen += batch_data.size(0)
        batch_data = batch_data.to(device)
        # Sample random Gaussian noise
        z = torch.randn(batch_data.size(0), 512).to(device)
        # Interpolate between target image histogram 
        # to prevent overfitting to dataset images
        # target_hist = None 
        target_hist = random_interpolate_hists(batch_data)
        # Generate fake images
        fake_data, w = generator(z, target_hist)
        # Detach fake data so no gradient accumalition 
        # to generator while only training discriminator
        fake_data = fake_data.detach()
        
        # Compute real probabilities computed by discriminator
        fake_scores = discriminator(fake_data)
        real_scores = discriminator(batch_data)
        
        # gradient_penalty1 = compute_gradient_penalty(fake_data, batch_data, discriminator)
        # batch_data.requires_grad_()
        # gradient_penalty2 = gradient_penalty(batch_data, real_scores)
        # d_loss = torch.nn.functional.relu(1+torch.mean(real_scores)) + torch.nn.functional.relu(1-torch.mean(fake_scores))
        real_loss = torch.mean(torch.nn.functional.softplus(-real_scores))/ acc_gradient_iter
        fake_loss = torch.mean(torch.nn.functional.softplus(fake_scores)) / acc_gradient_iter # torch.mean(fake_scores) /  acc_gradient_iter  #  
        d_loss = real_loss + fake_loss
        cumulative_dloss += d_loss.detach().item()/update_disc
        # d_loss = -torch.mean(real_scores) - 1 +torch.mean(fake_scores)
        # print("Grad diffs",abs((gradient_penalty1*10 - gradient_penalty2)).max())
        # in stylegan2 paper they argue applying regularization in every 16 iteration does not hurt perfrormance 
        if disc_step % 4 == 0: 
            # r1 regulatization
            # for autograd.grad to work input should also have requires_grad = True
            batch_data_grad = batch_data.clone().detach().requires_grad_(True)
            real_score_for_r1 = discriminator(batch_data_grad)
            # gradients1 = torch.autograd.grad(outputs=real_score_for_r1, inputs=batch_data_grad, grad_outputs=torch.ones(real_score_for_r1.size()).to(device))[0]
            # r1_reg = torch.mean(torch.sum(torch.square(gradients1.view(gradients1.size(0), -1)), dim=1))
            # gp = r1_factor*r1_reg
            gp = gradient_penalty(batch_data_grad, real_score_for_r1)
            cum_gp = gp.item()
            d_loss = d_loss + gp
        # print(r1_reg.size())
        # print((r1_reg-gradients1.norm(2, dim=1)))
        d_loss = d_loss/update_disc
        d_loss.backward()
        disc_optim.step()

        # print("%", training_percent, " Disc loss:", cumulative_dloss)
        if break_flag: 
            break_flag=False
            break
        # Update Generator
        gene_optim.zero_grad()
        cumulative_gloss = 0.0
        totalg = None
        # for _ in range(update_disc):
        z = torch.randn(batch_data.size(0), 512).to(device)
        fake_data, w = generator(z, target_hist) 
        # fake_data = torch.clamp(fake_data, -256, 256)
        
        disc_score = discriminator(fake_data)
        # g_loss = non_sat_generator_loss(fake_data, disc_score, target_hist) 
        # g_loss = torch.mean(disc_score)
        g_loss = non_sat_generator_loss(fake_data, disc_score, target_hist)
        if (disc_step+1) % 16 == 0:
            pl_noise = torch.randn_like(fake_data).to(device) / np.sqrt(fake_data.shape[2]*fake_data.shape[3])
            gradients2 = torch.autograd.grad(outputs=fake_data*pl_noise, inputs=w, grad_outputs=torch.ones(fake_data.size()).to(device), retain_graph=True)[0]
            j_norm  = torch.sqrt(torch.sum(torch.square(gradients2.view(gradients2.size(0), -1)),dim=1))
            if target_scale is None:
                target_scale = j_norm
            plr = torch.mean(torch.square(j_norm - target_scale))
            g_loss = g_loss + plr * plr_factor
            target_scale = (1-ema_decay_coeff)* target_scale + ema_decay_coeff * j_norm
        g_loss = g_loss/update_disc
        cumulative_gloss += g_loss.detach().item()/update_disc
        g_loss.backward()
        gene_optim.step()
        print("%", training_percent, "G:", cumulative_gloss, "D:", cumulative_dloss, "GP:", cum_gp)


        if (disc_step) % save_iter == 0:
            print("Saving img stats:", torch.min(fake_data), torch.max(fake_data), torch.mean(fake_data))
            fake_data = truncation_trick(generator, 512, batch_size, batch_data)
            # wandb.log({"generated_image": wandb.Image(fake_data[np.random.randint(0, fake_data.size(0)-1)])})
            save_image(fake_data[0], os.path.join(fake_image_dir, "fake_{}_{}.png".format(epoch, disc_step)), normalize=True)
            torch.save(generator.state_dict(), "generator_{}.pt".format(epoch))
            torch.save(discriminator.state_dict(), "discriminator_{}.pt".format(epoch))
        disc_step += 1 
        