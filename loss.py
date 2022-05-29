# implement loss functions
import utils
import torch

# See Eq. 5
# g: generated image, d_score: scalar output of discriminator
def generator_loss(g, d_score, hist_t):
    #c_loss = hellinger_dist_loss(g, hist_t)
    #alpha = 2.0  # See Sec. 5.2 Training details
    g_loss = torch.mean(torch.log(torch.sigmoid(d_score))) #- alpha*c_loss
    return -g_loss 

def disc_loss(g_scores, r_scores):
    return -torch.mean(torch.log(torch.sigmoid(r_scores))) - torch.mean(torch.log(1-torch.sigmoid(g_scores)))

# This is color matching loss, see Eq. 4
# It takes histogram of generated and target
def hellinger_dist_loss(g, hist_t, device="cuda"):
    hist_g = utils.histogram_feature_v2(g, device=device)  # Compute histogram feature of generated img
    t_sqred = torch.sqrt(hist_t)
    g_sqred = torch.sqrt(hist_g)
    diff = t_sqred - g_sqred
    h = torch.sum(torch.square(diff), dim=(1,2,3))
    # print(hist_t.min(), hist_g.min())
    h_norm = torch.sqrt(h)
    h_norm = h_norm * (torch.sqrt(torch.ones((g.shape[0]))/2))
    
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