# implement loss functions
from dis import dis
from matplotlib.pyplot import hist2d
import utils
import torch

# See Eq. 5
# g: generated image, d_score: scalar output of discriminator
def generator_loss(g, d_score):
    c_loss = hellinger_dist_loss(g)
    alpha = 2.0  # See Sec. 5.2 Training details
    g_loss = d_score + alpha*c_loss
    return -g_loss 

def disc_loss(g_scores, r_scores):
    return torch.mean(-torch.log(r_scores)) - torch.mean(torch.log(1-g_scores))

# This is color matching loss, see Eq. 4
# It takes histogram of generated and target
def hellinger_dist_loss(g):
    hist_g = utils.histogram_feature(g)  # Compute histogram feature of generated img

    # Either read two images and compute hists
    # Note loaded batch is 2*M hists
    # Or use precomputed hists
    # hist1 = torch.load()
    # hist2 = torch.load()

    # For testing the function
    hist1 = torch.rand((2, 3, 64, 64))
    hist2 = torch.rand((2, 3, 64, 64))

    delta = torch.rand((g.shape[0],1))
    delta = torch.unsqueeze(delta, dim=2)
    delta = torch.unsqueeze(delta, dim=3)
    
    hist_t = delta*hist1 + (1-delta)*hist2
    t_sqred = torch.square(hist_t)
    g_sqred = torch.square(hist_g)
    diff = t_sqred - g_sqred
    h_norm = torch.norm(torch.norm(torch.norm(diff, dim=3), dim=2), dim=1)
    h_norm = h_norm * (torch.sqrt(torch.ones((g.shape[0]))*2))
    
    # Used mean reduction, other option is sum reduction
    h_norm = torch.mean(h_norm)

    return h_norm

def main():
    g = torch.rand((2, 3, 64, 64))
    hellinger_dist_loss(g)
    g_scores = torch.rand(15)
    r_scores = torch.rand(19)
    print(disc_loss(g_scores, r_scores))

if __name__=="__main__": main()