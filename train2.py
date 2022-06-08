# implement training loop 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from torch_optimizer import DiffGrad
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN
from loss import wgan_gp_gen_loss, wgan_gp_disc_loss
from utils import histogram_feature_v2

from functools import partial
import os

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# def random_interpolate_hists(batch_data):
#     B = batch_data.size(0)
#     delta = torch.rand((B,1,1,1)).to(device)
#     first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
#     second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
#     first_hist = histogram_feature_v2(first_images)
#     second_hist = histogram_feature_v2(second_images)
#     hist_t = delta*first_hist + (1-delta)*second_hist
#     return hist_t

# def random_interpolate_hists(batch_data):
#     hist_list = []
#     hist = histogram_feature_v2(batch_data)
#     hist_t = delta*first_hist + (1-delta)*second_hist
#     return hist_t

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
    [transforms.Resize((64,64)),
     transforms.RandomHorizontalFlip(0.5)])
dataset = AnimeFacesDataset(real_image_dir, transform, device)

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epochs = 10

generator = HistoGAN()
discriminator = Discriminator(7, 16)

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
acc_gradient_iter = 16
r1_factor = 1

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
target_scale = torch.tensor([0]).to(device)  # Should not be 0, first weights should be used
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
update_disc = 16
dataiter = iter(dataloader)
print("Starting to train")
break_flag = False
for epoch in range(num_epochs):
    dataiter = iter(dataloader)
    
    while seen < len(dataset):
        torch.cuda.empty_cache()
        cumulative_dloss = 0.0
        disc_optim.zero_grad()
        if break_flag: 
            break_flag=False
            break
        for _ in range(update_disc): 
            training_percent = 100*seen/len(dataset)
            if seen+batch_size <= len(dataset): 
                batch_data = next(dataiter)
            else: 
                break_flag = True
                break
            seen += batch_data.size(0)
            batch_data = batch_data.to(device)
            # Sample random Gaussian noise
            z = torch.randn(batch_data.size(0), 512).to(device)
            # Interpolate between target image histogram 
            # to prevent overfitting to dataset images
            target_hist = None #random_interpolate_hists(batch_data)
            # Generate fake images
            fake_data, w = generator(z, target_hist)
            # fake_data = torch.clamp(fake_data, -256, 256)

            # Detach fake data so no gradient accumalition 
            # to generator while only training discriminator
            fake_data = fake_data.detach()

            # Compute real probabilities computed by discriminator
            fake_scores = discriminator(fake_data)
            real_scores = discriminator(batch_data)
            gradient_penalty = compute_gradient_penalty(fake_data, batch_data, discriminator)

            #d_loss = disc_loss(fake_scores, real_scores)
            d_loss = torch.nn.functional.relu(1+torch.mean(real_scores)) + torch.nn.functional.relu(1-torch.mean(fake_scores))
            # in stylegan2 paper they argue applying regularization in every 16 iteration does not hurt perfrormance 
            # if (disc_step+1) % 16 == 0: 
            #     # r1 regulatization
            #     # for autograd.grad to work input should also have requires_grad = True
            #     batch_data_grad = batch_data.clone().detach().requires_grad_(True)
            #     real_score_for_r1 = discriminator(batch_data_grad)
            #     gradients1 = torch.autograd.grad(outputs=real_score_for_r1, inputs=batch_data_grad, grad_outputs=torch.ones(real_score_for_r1.size()).to(device))[0]
            #     r1_reg = torch.mean(torch.sum(torch.square(gradients1.view(gradients1.size(0), -1)), dim=1))
            #     d_loss = d_loss + r1_factor*r1_reg  

            d_loss = d_loss/update_disc
            d_loss.backward()
            cumulative_dloss += d_loss.item()
        disc_optim.step()

        print("%", training_percent, " Disc loss:", cumulative_dloss)

        # Update Generator
        gene_optim.zero_grad()
        cumulative_gloss = 0.0
        totalg = None
        for _ in range(update_disc):
            training_percent = 100*seen/len(dataset)
            if seen+batch_size <= len(dataset): 
                batch_data = next(dataiter)
            else: 
                break_flag = True
                break
            seen += batch_data.size(0)
            batch_data = batch_data.to(device)
            z = torch.randn(batch_data.size(0), 512).to(device)
            fake_data, w = generator(z, target_hist) 
            # fake_data = torch.clamp(fake_data, -256, 256)

            disc_score = discriminator(fake_data)
            #g_loss = generator_loss(fake_data, disc_score, target_hist) 
            g_loss = torch.mean(disc_score)
            # if (disc_step+1) % 16 == 0:
            #     gradients2 = torch.autograd.grad(outputs=fake_data*torch.randn_like(fake_data).to(device), inputs=w, grad_outputs=torch.ones(fake_data.size()).to(device), retain_graph=True)[0]
            #     j_norm  = torch.sqrt(torch.sum(torch.square(gradients2.view(gradients2.size(0), -1)),dim=1))
            #     plr = torch.mean(torch.square(j_norm - target_scale))
            #     g_loss = g_loss + plr * plr_factor
            #     target_scale = (1-ema_decay_coeff)* target_scale + ema_decay_coeff * j_norm
            g_loss = g_loss/update_disc
            cumulative_gloss += g_loss.item()
            g_loss.backward()
        gene_optim.step()
        print("%", training_percent, "Gen loss:", g_loss.item())


        if (disc_step) % save_iter == 0:
            print("Saving img stats:", torch.min(fake_data), torch.max(fake_data), torch.mean(fake_data))
            
            # with open("gen_params.txt", "a") as f:
            #     for name, param in generator.named_parameters():
            #         f.write(name + " " + str(torch.mean(param)) + " " +str(torch.var(param)) + "\n")
            #     f.write("\n")
                    
            # with open("disc_params.txt", "a") as f:
            #     for name, param in generator.named_parameters():
            #         f.write(name + " " + str(torch.mean(param)) + " " +str(torch.var(param)) + "\n")
            #     f.write("\n")

            # tgen_params = []
            # tdisc_params = []

            # for name, param in generator.named_parameters():
            #     param = param.detach().cpu().numpy()
            #     tgen_params.append([np.mean(param), np.var(param)])
            # for name, param in discriminator.named_parameters():
            #     param = param.detach().cpu().numpy()
            #     tdisc_params.append([np.mean(param), np.var(param)])

            # tgen_params = np.array(tgen_params)
            # tdisc_params = np.array(tdisc_params)

            # dgen = abs(tgen_params - gen_params)
            # ddisc = abs(tdisc_params - disc_params)

            # print("generator differences:", np.max(dgen, axis=0), np.min(dgen, axis=0), np.mean(dgen, axis=0), np.var(dgen, axis=0))
            # print("disctiminator differences:", np.max(ddisc, axis=0), np.min(ddisc, axis=0), np.mean(ddisc, axis=0), np.var(ddisc, axis=0))

            for i in range(fake_data.size(0)):
                save_image(fake_data[i], os.path.join(fake_image_dir, "fake_{}_{}_{}.png".format(epoch, disc_step, i)))
            # torch.save(generator.state_dict(), "generator.pt")
            # torch.save(discriminator.state_dict(), "discriminator.pt")
        disc_step += 1 
        