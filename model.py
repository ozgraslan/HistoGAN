# implement histogan model
import torch
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearLayer(torch.nn.Module):
    def __init__(self, inp_size, out_size, lr_bias, kaiming_init=True, use_eqlr=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(inp_size, out_size))#.to(device)
        self.bias = torch.nn.Parameter(torch.randn(out_size))#.to(device)
        if kaiming_init:
            torch.nn.init.kaiming_normal_(self.weight)
        if use_eqlr:
            self.eqlr_coeff = lr_bias / torch.sqrt(torch.tensor([inp_size]))
            self.eqlr_coeff = self.eqlr_coeff.to(device)
        self.use_eqlr = use_eqlr

    def forward(self, x):
        weight = self.weight 
        if self.use_eqlr:
            weight*= self.eqlr_coeff 
        z = torch.addmm(self.bias, x, weight)
        return z

def convolution(inp, weight, padding, stride):
    # Implementaion of convolution operation using torch.unfold 
    # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # Changed such that it can work with batch of styles
    # After the operation the output feature maps' spatial size should be the same as the input, therefore padding=1
    inp_unf = torch.nn.functional.unfold(inp, (weight.size(-1), weight.size(-1)), padding=(padding, padding), stride=(stride, stride)) 
    if len(weight.size()) == 4:
        out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        outsize1 = int(1+(inp.size(2)+2*padding -weight.size(2))/stride) 
        outsize2 = int(1+(inp.size(3)+2*padding -weight.size(3))/stride) 
    else:
        out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), weight.size(1), -1).transpose(1,2)).transpose(1, 2)
        outsize1 = int(1+(inp.size(2)+2*padding -weight.size(3))/stride) 
        outsize2 = int(1+(inp.size(3)+2*padding -weight.size(4))/stride) 


    out = torch.nn.functional.fold(out_unf, (outsize1, outsize2), (1, 1))
    return out    

class ModDemodConv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, stride=1, moddemod=True, kaiming_init=True, use_eqlr=False):
        super().__init__()
        # StyleGAN papers mention each weight is initalized with N(0, I)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))#.to(device)
        if kaiming_init:
            torch.nn.init.kaiming_normal_(self.weight)
        self.padding = padding
        self.stride = stride
        self.moddemod = moddemod
        if not self.moddemod:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))#.to(device)
        else:
            self.bias = None
        
        if use_eqlr:
            self.eqlr_coeff = 1/torch.sqrt(torch.tensor([in_channels*3*3]))
            self.eqlr_coeff = self.eqlr_coeff.to(device)
        self.use_eqlr = use_eqlr

    def forward(self, x, style=None):
        weight = self.weight 
        if self.use_eqlr:
            weight *= self.eqlr_coeff
        if self.moddemod:
            style = style.view(style.size(0),1,-1,1,1)
            modulated_weight = style * weight
            sigma = torch.sqrt(modulated_weight.square().sum(dim=(2,3,4), keepdim=True) +1e-8)
            demodulated_weight = modulated_weight / sigma
            out = convolution(x, demodulated_weight, self.padding, self.stride)
        else:
            out = convolution(x, weight, self.padding, self.stride) 
            b,c,h,w = out.size()
            out = out + torch.permute(self.bias.repeat(b,h,w,1), (0,3,1,2))
        return out     

class ModConv1x1(torch.nn.Module):
    def __init__(self, in_channels, out_channels=3, padding=0, stride=1, mod=True, kaiming_init=True, use_eqlr=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))#.to(device)
        if kaiming_init:
            torch.nn.init.kaiming_normal_(self.weight)
        self.padding = padding
        self.stride = stride
        self.mod = mod
        if not self.mod:
            self.bias = torch.nn.Parameter(torch.randn(out_channels))#.to(device) 
        else:
            self.bias = None
        if use_eqlr:
            self.eqlr_coeff = 1/torch.sqrt(torch.tensor([in_channels]))
            self.eqlr_coeff = self.eqlr_coeff.to(device)
        self.use_eqlr = use_eqlr

    def forward(self, x, style=None):
        weight = self.weight 
        if self.use_eqlr:
            weight *= self.eqlr_coeff
        if self.mod:
            style = style.view(style.size(0),1,-1,1,1)
            modulated_weight = style * weight
            out = convolution(x, modulated_weight, self.padding, self.stride)
        else:
            out = convolution(x, weight, self.padding, self.stride) 
            b,c,h,w = out.size()
            out = out + torch.permute(self.bias.repeat(b,h,w,1), (0,3,1,2)) 
        return out 

class ResidualBlock(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, reduce_size=False, use_spec_norm=False):
        super().__init__()
        ## residual block architecture is taken from the paper Figure S1
        if use_spec_norm:
            self.conv1 = spectral_norm(torch.nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1, stride=1))    # ModDemodConv3x3(inp_dim, out_dim, padding=1, stride=1, moddemod=False)  
            self.conv2 = spectral_norm(torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1))   # ModDemodConv3x3(out_dim, out_dim, padding=1, stride=1, moddemod=False)  
            self.conv1x1 = spectral_norm(torch.nn.Conv2d(inp_dim, out_dim, kernel_size=1, padding=0, stride=1))  # ModConv1x1(inp_dim, out_dim, padding=0, stride=1, mod=False)         
        else:
            self.conv1 = torch.nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1, stride=1)   # ModDemodConv3x3(inp_dim, out_dim, padding=1, stride=1, moddemod=False)  
            self.conv2 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1)  # ModDemodConv3x3(out_dim, out_dim, padding=1, stride=1, moddemod=False)  
            self.conv1x1 =torch.nn.Conv2d(inp_dim, out_dim, kernel_size=1, padding=0, stride=1) # ModConv1x1(inp_dim, out_dim, padding=0, stride=1, mod=False)         
        self.lrelu = torch.nn.LeakyReLU(0.2)
        if reduce_size:
            self.conv3 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2)   # ModDemodConv3x3(out_dim, out_dim, padding=1, stride=2, moddemod=False)  
        else:
            self.conv3 = None

    def forward(self, x):
        res = self.conv1x1(x) 
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out += res
        if not self.conv3 is None:
            out = self.conv3(out)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, network_capacity=16, image_res=256, kaiming_init=True, use_spec_norm=False):
        super().__init__()
        ## Discriminator architecture taken from Section 5.1. Details of Our Networks
        residual_block_layers = []
        in_dim = 3
        out_dim = network_capacity
        num_res_block = int(np.log2(image_res))
        for i in range(num_res_block):
            reduce_size = i != (num_res_block-1)
            residual_block_layers.append(ResidualBlock(in_dim, out_dim, reduce_size, use_spec_norm))
            # if reduce_size:
            #     if use_spec_norm:
            #         residual_block_layers.append(spectral_norm(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2 ,padding=1)))
            #     else:
            #         residual_block_layers.append(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2 ,padding=1))
            # else:
            #     if use_spec_norm:
            #         residual_block_layers.append(spectral_norm(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)))
            #     else:
            #         residual_block_layers.append(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1))
            in_dim = out_dim
            out_dim = 2*out_dim 
        self.residual_layers = torch.nn.Sequential(*residual_block_layers)
        linear_inp = 2*2*in_dim
        self.fc = torch.nn.Linear(linear_inp, 1)
        if kaiming_init:
            for mod in self.modules():
                if type(mod) in [torch.nn.Linear, torch.nn.Conv2d]:
                    torch.nn.init.kaiming_normal_(mod.weight)
                
    def forward(self, x):
        out = self.residual_layers(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class StyleGAN2Block(torch.nn.Module):
    # StyleGan2 Block depicted in Figure 2/A of HistoGAN paper
    def __init__(self, input_channel_size, output_channel_size, latent_size=512, kaiming_init=True, use_eqlr=False): # these channel numbers should be decided soon
        super().__init__()
        self.affine_1 = torch.nn.Linear(latent_size, input_channel_size)
        self.affine_2 = torch.nn.Linear(latent_size, output_channel_size)
        self.affine_3 = torch.nn.Linear(latent_size, output_channel_size)
        # In stylegan2 paper, section B. Implementation Details/Generator redesign  
        # authors say that they used only single shared scaling factor for each feature map 
        # and they initiazlize the factor by 0
        self.noise_scaling_factor_1 = torch.nn.Parameter(torch.zeros((1)))
        self.noise_scaling_factor_2 = torch.nn.Parameter(torch.zeros((1))) 
        self.md3x3_1 = ModDemodConv3x3(input_channel_size, output_channel_size,padding=1, kaiming_init=True, use_eqlr=False)
        self.md3x3_2 = ModDemodConv3x3(output_channel_size, output_channel_size, padding=1, kaiming_init=True, use_eqlr=False)
        self.m1x1 = ModConv1x1(output_channel_size, 3, kaiming_init=True, use_eqlr=False)
        self.bias_1 = torch.nn.Parameter(torch.zeros(output_channel_size))
        self.bias_2 = torch.nn.Parameter(torch.zeros(output_channel_size))

        self.lrelu = torch.nn.LeakyReLU(0.2)


    def forward(self, fm, w):
        # fm: input feature map (B, C, H, W), C = channel1
        # latent vector w, w = f(z), (B, 512)
        # in figure noise vectors are outside the block
        # but it does no matter since they are independent and random
        # in style gan2 paper each noise is tought as a N(0,I) image 
        # with same height and with as the feature map
        # print("noise scaling factors:",self.noise_scaling_factor_1, self.noise_scaling_factor_2)
        batch, channel, height, width = fm.size()  
        noise_1 = torch.randn((batch, 1, height, width)).to(device)
        noise_2 = torch.randn((batch, 1, height, width)).to(device)
        style_1 = self.affine_1(w)
        style_2 = self.affine_2(w)
        style_3 = self.affine_3(w)
        # if bias is not reshaped gives error, this version can broadcast to batch
        # print(self.md3x3_1(fm, style_1).size(), noise_1.size())
        out = self.md3x3_1(fm, style_1) + self.noise_scaling_factor_1 * noise_1 + self.bias_1.view(-1,1,1) 
        out = self.lrelu(out)
        out = self.md3x3_2(out, style_2) + self.noise_scaling_factor_2 * noise_2 + self.bias_2.view(-1,1,1)
        out = self.lrelu(out)
        rgb_out = self.m1x1(out, style_3) # rgb_out should then be upsampled, for now left it out of this block
        return out, rgb_out

class StyleGAN2Block2(torch.nn.Module):
    # StyleGan2 Block depicted in Figure 2/A of HistoGAN paper
    def __init__(self, input_channel_size, output_channel_size, latent_size=512): # these channel numbers should be decided soon
        super().__init__()
        self.affine_1 = torch.nn.Linear(latent_size, input_channel_size)
        self.affine_2 = torch.nn.Linear(latent_size, output_channel_size)
        self.affine_3 = torch.nn.Linear(latent_size, output_channel_size)
        # In stylegan2 paper, section B. Implementation Details/Generator redesign  
        # authors say that they used only single shared scaling factor for each feature map 
        # and they initiazlize the factor by 0
        self.noise_scaling_factor_1 = torch.nn.Parameter(torch.zeros((1)))
        self.noise_scaling_factor_2 = torch.nn.Parameter(torch.zeros((1))) 
        self.conv_1 = torch.nn.Conv2d(input_channel_size, output_channel_size, kernel_size=3, padding=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(output_channel_size, output_channel_size, kernel_size=3, padding=1, bias=False)
        self.conv1x1 = torch.nn.Conv2d(output_channel_size, 3, kernel_size=1, bias=False)
        self.bias_1 = torch.nn.Parameter(torch.zeros(output_channel_size))
        self.bias_2 = torch.nn.Parameter(torch.zeros(output_channel_size))

        self.lrelu = torch.nn.LeakyReLU(0.2)


    def forward(self, fm, w):
        # fm: input feature map (B, C, H, W), C = channel1
        # latent vector w, w = f(z), (B, 512)
        # in figure noise vectors are outside the block
        # but it does no matter since they are independent and random
        # in style gan2 paper each noise is tought as a N(0,I) image 
        # with same height and with as the feature map
        # print("noise scaling factors:",self.noise_scaling_factor_1, self.noise_scaling_factor_2)
        batch, channel, height, width = fm.size()  
        noise_1 = torch.randn((batch, 1, height, width)).to(device)
        noise_2 = torch.randn((batch, 1, height, width)).to(device)
        style_1 = self.affine_1(w)
        style_2 = self.affine_2(w)
        style_3 = self.affine_3(w)
        out = fm * style_1.view(batch, -1, 1, 1)
        out = self.conv_1(out)
        out = out/torch.std(out, unbiased=False)
        out = out + self.noise_scaling_factor_1 * noise_1 + self.bias_1.view(-1, 1, 1)
        out = self.lrelu(out)
        out = out * style_2.view(batch, -1, 1, 1)
        out = self.conv_2(out)
        out = out/torch.std(out, unbiased=False)
        out = out + self.noise_scaling_factor_2 * noise_2 + self.bias_2.view( -1, 1, 1)
        out = self.lrelu(out)
        rgb_out = self.conv1x1(out * style_3.view(batch, -1, 1, 1))
        return out, rgb_out

class HistoGAN(torch.nn.Module):
    def __init__(self, network_capacity=16, latent_size=512, h=64, image_res=256, mapping_layer_num=8, kaiming_init=True, use_eqlr=False):
        super().__init__()
        latent_mapping_list = []
        # hist_projection_list = []
        # hist_projection_inp = h*h*3
        num_gen_layers = int(np.log2(image_res)-1)
        for i in range(mapping_layer_num):
            # if i == (mapping_layer_num-1):
            #     latent_mapping_list.append(torch.nn.Linear(latent_size, latent_size))
                #hist_projection_list.append(LinearLayer(latent_size, latent_size, 0.01, kaiming_init=True, use_eqlr=False))
            # else:
            latent_mapping_list.extend([torch.nn.Linear(latent_size, latent_size),
                                        torch.nn.LeakyReLU(0.2, inplace=True)])

                #hist_projection_list.extend([LinearLayer(hist_projection_inp, latent_size, 0.01, kaiming_init=True, use_eqlr=False), torch.nn.LeakyReLU(0.2)])
                #hist_projection_inp  = latent_size

        self.latent_mapping = torch.nn.Sequential(*latent_mapping_list)
        #self.hist_projection = torch.nn.Sequential(*hist_projection_list)
        self.learned_const_inp = torch.nn.Parameter(torch.randn(4*network_capacity, 4, 4))
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        stylegan2_block_list = [] 
        inp_size = 4 * network_capacity
        out_size = network_capacity * 2**num_gen_layers
        for _ in range(num_gen_layers):
            # print(inp_size, out_size)
            stylegan2_block_list.append(StyleGAN2Block2(inp_size, out_size, latent_size=latent_size))
            inp_size = out_size
            out_size //= 2
        self.stylegan2_blocks = torch.nn.ModuleList(stylegan2_block_list)
        if kaiming_init:
            for mod in self.modules():
                if type(mod) in [torch.nn.Linear, torch.nn.Conv2d]:
                    torch.nn.init.kaiming_normal_(mod.weight)

    def forward(self, z, target_hist):
        # noise input z, size: B, latent_size
        B = z.size(0)
        w = self.latent_mapping(z)
        fm = self.learned_const_inp.expand(B, -1, -1, -1)
        for i, stylegan2_block in enumerate(self.stylegan2_blocks): # [:-1]
            fm, rgb = stylegan2_block(fm, w[:,i,:])
            fm = self.upsample(fm)
            if i == 0:
                rgb = self.upsample(rgb)
                rgb_sum = rgb
            elif i == len(self.stylegan2_blocks)-1:
                rgb_sum += rgb
            else:
                rgb_sum += rgb
                rgb_sum = self.upsample(rgb_sum)
        #hist_w  = self.hist_projection(target_hist.flatten(1))
        #_, rgb = self.stylegan2_blocks[-1](fm, hist_w)
        #rgb_sum += rgb
        # w is returned to compute path length regularization
        return rgb_sum, w 

    def get_w_from_z(self, z):
        w = self.latent_mapping(z)
        return w

    def gen_image_from_w(self, w, target_hist):
        B = w.size(0)
        # print(w.size())
        fm = self.learned_const_inp.expand(B, -1, -1, -1)
        for i, stylegan2_block in enumerate(self.stylegan2_blocks): # [:-1]
            fm, rgb = stylegan2_block(fm, w[:,i,:])
            fm = self.upsample(fm)
            if i == 0:
                rgb = self.upsample(rgb)
                rgb_sum = rgb
            elif i == len(self.stylegan2_blocks) -1:
                rgb_sum += rgb
            else:
                rgb_sum += rgb
                rgb_sum = self.upsample(rgb_sum)
        # hist_w  = self.hist_projection(target_hist.flatten(1))
        # _, rgb = self.stylegan2_blocks[-1](fm, hist_w)
        # rgb_sum += rgb
        return rgb_sum


    