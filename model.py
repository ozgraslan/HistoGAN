# implement histogan model
import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        ## residual block architecture is taken from the paper Figure S1
        self.conv1 = torch.nn.Conv2d(inp_dim, out_dim, kernel_size=3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=1)
        self.conv1x1 = torch.nn.Conv2d(inp_dim, out_dim, kernel_size=1, padding=0, stride=1)
        self.conv3 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2)
        self.lrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out += self.conv1x1(x)
        out = self.conv3(out)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, num_res_block, output_channels):
        super().__init__()
        ## Discriminator architecture taken from Section 5.1. Details of Our Networks
        residual_block_layers = []
        in_dim = 3
        for _ in range(num_res_block):
            residual_block_layers.append(ResidualBlock(in_dim, output_channels))
            in_dim = output_channels
            output_channels = 2*output_channels 
        self.residual_layers = torch.nn.Sequential(*residual_block_layers)

        ## test tensor is used to calculate the input dimension to the fc layer
        test =  torch.zeros((3,256,256))
        with torch.no_grad():
            out = self.residual_layers(test)
        linear_inp = out.size(0) * out.size(1) * out.size(2)

        self.fc = torch.nn.Linear(linear_inp, 1)
    
    def forward(self, x):
        out = self.residual_layers(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ModDemodConv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # StyleGAN papers mention each weight is initalized with N(0, I)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        # after the operation the output feature maps' spatial size should be the same as the input
        self.padding = 1  

    def convolution(self, inp, weight):
        # Implementaion of convolution operation using torch.unfold 
        # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # Changed such that it can work with batch of styles
        inp_unf = torch.nn.functional.unfold(inp, (3, 3), padding=(1,1))
        out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), weight.size(1), -1).transpose(1, 2)).transpose(1, 2)
        out = torch.nn.functional.fold(out_unf, (inp.size(2), inp.size(3)), (1, 1))
        return out    
    
    def forward(self, x, style):
        style = style.view(style.size(0),1,-1,1,1)
        modulated_weight = style * self.weight
        # print(modulated_weight.size())
        sigma = torch.sqrt(modulated_weight.square().sum(dim=(2,3,4), keepdim=True) +1e-3)
        # print(sigma.size())
        demodulated_weight = modulated_weight / sigma
        # print(demodulated_weight.size())
        out = self.convolution(x, demodulated_weight)
        return out     