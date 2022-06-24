# implement loss functions
import utils
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# See Eq. 5
# g: generated image, d_score: scalar output of discriminator
def non_sat_generator_loss(g, d_score, hist_t):
    c_loss = hellinger_dist_loss(g, hist_t)
    alpha = 2.0  # See Sec. 5.2 Training details
    # print(alpha*c_loss)
    g_loss = torch.mean(d_score) + alpha*c_loss
    # g_loss = torch.mean(torch.nn.functional.softplus(-d_score)) + alpha*c_loss
    return g_loss 

def classics_disc_loss(g_scores, r_scores):
    return -torch.mean(torch.log(torch.sigmoid(r_scores))) - torch.mean(torch.log(1-torch.sigmoid(g_scores)))

def wgan_gp_gen_loss(disc_score):
    return -torch.mean(disc_score)

def wgan_gp_disc_loss(real_scores, fake_scores, gradient_penalty, coeff_penalty):
    return -torch.mean(real_scores) + torch.mean(fake_scores) + coeff_penalty*gradient_penalty

def gp_only_real(real_data, real_scores, r1_factor):
    # for autograd.grad to work input should also have requires_grad = True
    # print(real_scores.size(), real_data.size())
    gradients = torch.autograd.grad(outputs=real_scores.mean(), inputs=real_data, create_graph=True, retain_graph=True)[0]
    # print(gradients.size())
    gradients = gradients.view(real_data.size(0), -1)
    gradient_norm = torch.sqrt(1e-8+torch.sum(torch.square(gradients), dim=1))
    r1_reg = torch.mean(torch.square(gradient_norm-1))
    return r1_factor*r1_reg  

def pl_reg(generator, target_hist, target_scale, plr_factor, ema_decay_coeff):
    z = torch.randn(10, 512).to(device)
    fake_data, w = generator(z, target_hist)
    y = torch.randn_like(fake_data) / np.sqrt(fake_data.size(2) * fake_data.size(3))
    y = y.to(device)
    gradients = torch.autograd.grad(outputs=(fake_data*y).sum(), inputs=w, create_graph=True)[0]
    j_norm  = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
    j_norm_mean = torch.mean(j_norm)
    target_scale = (1-ema_decay_coeff)* target_scale + ema_decay_coeff * j_norm_mean.item()
    plr = torch.square(j_norm - target_scale)
    pl_reg = plr * plr_factor
    return torch.mean(pl_reg), target_scale

# This is color matching loss, see Eq. 4
# It takes histogram of generated and target
def hellinger_dist_loss(g, hist_t):
    relu_g = torch.nn.functional.relu(g, inplace=False)
    relu_g = torch.clamp(relu_g, 0, 1) # Fix relu inplace gradient
    hist_g = utils.histogram_feature_v2(relu_g)  # Compute histogram feature of generated img
    t_sqred = torch.sqrt(hist_t)
    # print("Target", torch.isnan(t_sqred))
    g_sqred = torch.sqrt(hist_g)
    # print("Gen", hist_g)
    diff = t_sqred - g_sqred
    h = torch.sum(torch.square(diff), dim=(1,2,3))
    # print(hist_t.min(), hist_g.min())
    h_norm = torch.sqrt(h)
    h_norm = h_norm * (torch.sqrt(torch.ones((g.shape[0]))/2).to(device))
    
    # Used mean reduction, other option is sum reduction
    h_norm = torch.mean(h_norm)

    return h_norm

def compute_gradient_penalty(fake_data, real_data, discriminator):
    a = torch.rand((fake_data.size(0), 1, 1, 1)).to(device)
    comb_data = a* fake_data + (1-a)*real_data
    comb_data = comb_data.requires_grad_(True)
    comb_score = discriminator(comb_data)
    gradients = torch.autograd.grad(outputs=comb_score, inputs=comb_data, grad_outputs=torch.ones(comb_score.size()).to(device), create_graph=True, retain_graph=True)[0]
    gradient_norm = torch.sqrt(1e-8+torch.sum(torch.square(gradients.view(gradients.size(0), -1)), dim=1))
    gradient_penalty = torch.mean(torch.square(gradient_norm-1))
    return gradient_penalty

def gradient_penalty(images, output, weight=10):
  batch_size = images.shape[0]
  gradients = torch.autograd.grad(outputs=output, inputs=images,
                         grad_outputs=torch.ones(output.size()).cuda(),
                         create_graph=True, retain_graph=True,
                         only_inputs=True)[0]
  gradients = gradients.reshape(batch_size, -1)
  return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
