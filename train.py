# implement training loop 
from pkg_resources import run_script
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch_optimizer import DiffGrad
from data import AnimeFacesDataset
from model import Discriminator, HistoGAN
from loss import generator_loss, disc_loss
from utils import histogram_feature_v2

torch.autograd.set_detect_anomaly(True)
def random_interpolate_hists(batch_data):
    B = batch_data.size(0)
    delta = torch.rand((B,1,1,1)) 
    first_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)))
    second_images = torch.index_select(batch_data, dim=0, index=torch.randint(0, B, (B,)))
    first_hist = histogram_feature_v2(first_images)
    second_hist = histogram_feature_v2(second_images)
    hist_t = delta*first_hist + (1-delta)*second_hist
    return hist_t

device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"  # Test for ram usage, histogram is computed on cpu, fix it
image_dir = "images"
transfrom = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.RandomHorizontalFlip(0.5)])
dataset = AnimeFacesDataset(image_dir, transfrom, device)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epochs = 100
generator = HistoGAN().to(device)
discriminator = Discriminator(7, 16).to(device)

g_update_iter = 5

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
for epoch in range(num_epochs):
    for iter, batch_data in enumerate(dataloader):
        disc_optim.zero_grad()
        # Sample random Gaussian noise
        z = torch.rand(batch_data.size(0), 512)

        # Interpolate between target image histogram 
        # to prevent overfitting to dataset images
        target_hist = random_interpolate_hists(batch_data)
        # Generate fake images
        fake_data = generator(z, target_hist)
        # Detach fake data so no gradient accumalition 
        # to generator while only training discriminator
        fake_data = fake_data.detach()
        # Compute real probabilities computed by discriminator
        fake_scores = discriminator(fake_data)
        real_scores = discriminator(batch_data)
        # Compute discriminator loss
        loss = disc_loss(fake_scores, real_scores)
        print("Disc loss:", loss.item())
        loss.backward()
        disc_optim.step()

        if iter % g_update_iter == 0:
            gene_optim.zero_grad()
            print("hello")
            fake_data = generator(z, target_hist) 
            disc_score = discriminator(fake_data)
            print("hello2")
            loss = generator_loss(fake_data, disc_score, target_hist) 
            print("Gen loss:", loss.item())
            loss.backward()
            gene_optim.step()
            





