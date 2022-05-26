# implement training loop 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import AnimeFacesDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = "images"
transfrom = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.RandomHorizontalFlip(0.5)])
dataset = AnimeFacesDataset(image_dir, transfrom, device)

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)