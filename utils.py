# implement utility functions for image processing
from matplotlib.pyplot import hist
import numpy as np
import torch

# Generation of histogram origined from color constancy
# Paper: Jonathan T. Barron, Convolutional Color Constancy, 2015, https://arxiv.org/pdf/1507.00410.pdf
# given, RGB pixel value I is 
# the product of the “true” white-balanced RGB valueW for that pixel 
# and the RGB illumination L shared by all pixels
# aim to find W
# I = W x L
# In log space W = I - L
# hist_boundary taken from original code, 
# they are not mentioned in the paper.
# Not mentioned in the paper but "sampling" is used to get img pixels of size h

def histogram_feature(img, h=64, hist_boundary=[-3, 3], fall_off = 0.02):
    img_flat = torch.reshape(img, (img.shape[0], img.shape[1], -1))  # reshape such that img_flat = M,3,N*N
    
    # Pixel intesities I_y at Eq. 2
    eps = 1e-6  # prevent taking log of 0 valued pixels
    i_y = torch.sqrt(torch.square(img_flat[:, 0]) + torch.square(img_flat[:, 1]) + torch.square(img_flat[:, 2]))
    img += eps
    log_r = torch.log(img_flat[:, 0])
    log_g = torch.log(img_flat[:, 1])
    log_b = torch.log(img_flat[:, 2])
    
    # u,v parameters for each channel
    # each channel normalization values with respect to other two channels
    ur = log_r - log_g
    vr = log_r - log_b
    ug = -ur
    vg = -ur + vr
    ub = -vr
    vb = -vr + ur

    u = torch.linspace(hist_boundary[0], hist_boundary[1], h)
    u = torch.unsqueeze(u, dim=0)   # Make (h,) to (1, h) so that 
                                    # for each element in ur there will 
                                    # be difference with each u value.
    v = torch.linspace(hist_boundary[0], hist_boundary[1], h)
    v = torch.unsqueeze(v, dim=0)

    ur = torch.unsqueeze(ur, dim=2) # Make each element an array it 
                                    # Difference will be [N*N, 1] - [1,h] = [N*N, h]
                                    # See broadcasting for further
    ug = torch.unsqueeze(ug, dim=2) 
    ub = torch.unsqueeze(ub, dim=2)
    vr = torch.unsqueeze(vr, dim=2) 
    vg = torch.unsqueeze(vg, dim=2) 
    vb = torch.unsqueeze(vb, dim=2) 
    
    rdiffu = torch.abs(ur - u)
    gdiffu = torch.abs(ug - u)
    bdiffu = torch.abs(ub - u)
    rdiffv = torch.abs(vr - v)
    gdiffv = torch.abs(vg - v)
    bdiffv = torch.abs(vb - v)

    # So far for k Eq. 3, 
    # Inner absolute values for each pixel log-chrominance value with each bin of H(u,v,c)
    # k has two parts multiplied thus implemented as k = k_u x k_v
    rk_u = 1 / (1 + torch.square(rdiffu/fall_off))
    gk_u = 1 / (1 + torch.square(gdiffu/fall_off))
    bk_u = 1 / (1 + torch.square(bdiffu/fall_off))

    rk_v = 1 / (1 + torch.square(rdiffv/fall_off))
    gk_v = 1 / (1 + torch.square(gdiffv/fall_off))
    bk_v = 1 / (1 + torch.square(bdiffv/fall_off))

    rk_u = torch.unsqueeze(rk_u, dim=3)
    rk_v = torch.unsqueeze(rk_v, dim=2)
    rk = torch.matmul(rk_u, rk_v)
    
    gk_u = torch.unsqueeze(gk_u, dim=3)
    gk_v = torch.unsqueeze(gk_v, dim=2)
    gk = torch.matmul(gk_u, gk_v)
    
    bk_u = torch.unsqueeze(bk_u, dim=3)
    bk_v = torch.unsqueeze(bk_v, dim=2)
    bk = torch.matmul(bk_u, bk_v)

    i_y = torch.unsqueeze(i_y, dim=2)
    i_y = torch.unsqueeze(i_y, dim=3)

    # For each channel of H(u,v,c) = H(u,v,R), H(u,v,G), H(u,v,B), k values are computed above
    weighted_kr = rk*i_y  # Compute intensity weighted impact of chrominance values
    weighted_kg = gk*i_y  # Compute intensity weighted impact of chrominance values
    weighted_kb = bk*i_y  # Compute intensity weighted impact of chrominance values
    
    weighted_k = torch.stack([weighted_kr, weighted_kg, weighted_kb], dim=2)
    histogram = torch.sum(weighted_k, dim=1)
    sum_of_uvc = torch.sum(torch.sum(torch.sum(histogram, dim=3), dim=2), dim=1)
    sum_of_uvc = torch.reshape(sum_of_uvc, (-1, 1, 1, 1))
    histogram = histogram / sum_of_uvc
    
    return histogram

def main():
    batch = torch.rand((2, 3, 32, 32))
    # img1 = torch.stack((r,g,b), dim=0)
    # img2 = torch.stack((r,g,b), dim=0)
    # batch = torch.stack((img1, img2), dim=0)


if __name__=="__main__": main()