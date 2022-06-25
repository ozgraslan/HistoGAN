# implement evaluation functions
from multiprocessing import reduction
from matplotlib.image import imsave
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from data import AnimeFacesDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import histogram_feature_v2, truncation_trick, mixing_noise
from loss import hellinger_dist_loss
from model import HistoGAN, HistoGANAda
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
from torchvision.transforms import Resize

config = dict(
num_epochs = 10, # number of epochs for training
batch_size = 16, # batch size
acc_gradient_total = 16, # total number of samples seen by the networks in 1 iteration
r1_factor = 10, # coefficient of the r1 regularization term
r1_update_iter = 4, # in every r1_update_iter r1 regularization is used
decay_coeff = 0.99, # ema decay coefficient for updating the path length target varaible
plr_update_iter = 32, # in every plr_update_iter the path length regularization is used
save_iter = 400, # in every save_iter the images are saved
image_res = 64, # the resolution of the images
network_capacity = 16, # capacity of the network used for channels of constant input in generator 
latent_dim = 512, # dimensionalty of the noises
bin_size = 64, # bin size of the histograms
learning_rate = 0.0002, # learning rate
mapping_layer_num = 8, # number of Linear layers in Mapping part of the Generator (z -> w)
mixing_prob = 0.9, # probality of using two distinct noises for generation
use_plr = True, # Wheter to use path length reg in training
use_r1r = True, # Wheter to use r1 reg in training
kaiming_init = True, # Initiazlize networks with kaiming initialization method by He et al.
use_eqlr = False, # use eqularized learning coefficients for weights (similar to kaiming but used in every forward calculation)
use_spec_norm = False, # use spectral normalization of Discriminator weights (For stabilization)
disc_arch= "ResBlock", # architecture of the Discriminator (used for bookkeeping)
gen_arch = "InputModDemod", # architecture of the Generator (used for bookkeeping)
optim="Adam",  # Optimizer used (Adam or DiffGrad)
loss_type="wasser", # Loss type to use (Wasserstein, Hinge, Log Sigmoid)
pre_gen_name = None, # for loading a pretrained network
pre_disc_name = None, # for loading a pretrained network
)

# set global variables from config
device = "cuda" if torch.cuda.is_available() else "cpu"
real_image_dir = "images"
image_res = config["image_res"]
transform = transforms.Compose(
        [transforms.Resize((image_res,image_res))])
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
acc_gradient_total = config["acc_gradient_total"]
acc_gradient_iter = acc_gradient_total //batch_size
r1_factor = config["r1_factor"]
r1_update_iter = config["r1_update_iter"]
decay_coeff = config["decay_coeff"]
target_scale = torch.tensor([0], requires_grad=False).to(device)
plr_factor = np.log(2)/(256**2*(np.log(256)-np.log(2)))
plr_update_iter = config["plr_update_iter"]
save_iter = config["save_iter"]
network_capacity = config["network_capacity"] 
latent_dim = config["latent_dim"]
bin_size = config["bin_size"]
learning_rate = config["learning_rate"]
mapping_layer_num = config["mapping_layer_num"]
mixing_prob = config["mixing_prob"]
num_gen_layers = int(np.log2(image_res)-1)
use_plr = config["use_plr"]
use_r1r = config["use_r1r"]
kaiming_init= config["use_r1r"]
use_eqlr = config["use_eqlr"]
use_spec_norm = config["use_spec_norm"]
loss_type = config["loss_type"]
optim = config["optim"]
pre_gen_name = config["pre_gen_name"]
pre_disc_name = config["pre_disc_name"]
log_interval = 500

def random_interpolate_hists(batch_data, device="cpu"):
    B = batch_data.size(0)
    delta = torch.rand((B,1,1,1)).to(device)
    first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)).to(device))
    first_hist = histogram_feature_v2(first_images, device=device)
    second_hist = histogram_feature_v2(second_images, device=device)
    hist_t = delta*first_hist + (1-delta)*second_hist
    return hist_t

# Device "cpu" is advised by torch metrics
def fid_scores(generator, test_path, fid_batch=8, device="cpu"):
    transform = transforms.Compose([transforms.Resize((image_res, image_res))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=fid_batch, shuffle=True)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True)
    
    fids = []
    num_generated = 0
    for batch_data in dataloader:
        z = mixing_noise(fid_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        batch_data = batch_data*255
        fake_data = fake_data*255
        fake_data = fake_data.clamp(0, 255)
        batch_data = batch_data.byte()  # Convert to uint8 for fid
        fake_data = fake_data.byte()
        fid.update(batch_data, real=True)
        fid.update(fake_data, real=False)
        batch_fid = fid.compute()
        fids.append(batch_fid.item())
        num_generated += fid_batch
        if num_generated > 1000: break

    return fids

def hist_uv_kl(generator, test_path, kl_batch=8, device="cpu"):
    transform = transforms.Compose([transforms.Resize((image_res,image_res))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=kl_batch, shuffle=True)
    
    kls = []
    num_generated = 0
    for batch_data in dataloader:
        z = mixing_noise(kl_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        relu_fake = torch.nn.functional.relu(fake_data, inplace=False)
        relu_fake = torch.clamp(relu_fake, 0, 1) # Fix relu inplace gradient
        fake_hist = histogram_feature_v2(relu_fake, device=device)
        target_hist /= torch.linalg.norm(target_hist)
        fake_hist /= torch.linalg.norm(fake_hist)
        kl = (target_hist*(target_hist.log()-fake_hist.log())).sum()/kl_batch  # Compute KL Div
        kls.append(kl.detach().numpy())
        num_generated += kl_batch
        if num_generated > 1000: break
    
    return kls

def hist_uv_h(generator, test_path, h_batch=8, device="cpu"):
    transform = transforms.Compose([transforms.Resize((image_res,image_res))])
    dataset = AnimeFacesDataset(test_path, transform, device)
    dataloader = DataLoader(dataset, batch_size=h_batch, shuffle=True)
    
    hs = []
    num_generated = 0
    for batch_data in dataloader:
        z = mixing_noise(h_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
        target_hist = random_interpolate_hists(batch_data)
        fake_data, _ = generator(z, target_hist, test=True)
        h = hellinger_dist_loss(fake_data, target_hist, device=device)
        hs.append(h.detach().numpy())       
        num_generated += h_batch
        if num_generated > 1000: break
    
    return hs

def interpret(generator, color_img_path, device="cpu"):
    h_batch = 4
    transform = Resize((image_res, image_res))
    color_img = read_image(color_img_path).to(device).float()
    color_img = color_img/255.0
    color_img = transform(color_img)
    color_img = color_img.unsqueeze(dim=0)

    color_img = color_img.repeat((h_batch, 1, 1, 1))
    z = mixing_noise(h_batch, num_gen_layers, latent_dim, mixing_prob).to(device) 
    target_hist = histogram_feature_v2(color_img, device=device)
    fake_data, _ = generator(z, target_hist, test=True)
    # fake_data = truncation_trick(generator, latent_dim, h_batch, target_hist)
    save_image(fake_data, "fake_hist1.png")
    print("Saved")


def main():
    generator = HistoGAN(network_capacity, latent_dim, bin_size, image_res, mapping_layer_num, kaiming_init=kaiming_init, use_eqlr=use_eqlr)
    generator.load_state_dict(torch.load("/home/kovan-beta/GAN/HistoGan/HistoGAN/models/generator_9.pt"))
    
    # fids = fid_scores(generator, "/home/kovan-beta/GAN/HistoGan/HistoGAN/images", device="cpu")
    # fid_np = np.array(fids)
    # print(fid_np.mean())

    # kls = hist_uv_kl(generator, "images")
    # kl_np = np.array(kls)
    # print(kl_np.mean())

    # hist_dists = hist_uv_h(generator, "/home/kovan-beta/GAN/HistoGan/HistoGAN/images", device="cpu")
    # hd_np = np.array(hist_dists)
    # print(hd_np.mean())

    interpret(generator, "/home/kovan-beta/GAN/HistoGan/HistoGAN/yellow.jpg")
    # color_img = read_image("/home/kovan-beta/GAN/HistoGan/HistoGAN/yellow.jpg").to(device).float()
    # color_img = color_img/255.0 
    # save_image(color_img, "color.png")

if __name__=="__main__": main()