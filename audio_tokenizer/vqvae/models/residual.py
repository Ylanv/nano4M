import torch.nn as nn
import torch.nn.functional as F

""" Residual blocks 
Encoder/Decoder  : 2 Residual 3x3 blocks (ReLu,3x3 conv, ReLu, 1x1 conv) 256 hidden units
"""


class ResidualBlock(nn.Module):
    """Residual block:

    Parameters:
        in_dim (int) : the input dimension
        h_dim (int): the hidden layer dimension
        res_h_dim (int) : the hidden dimension of the residual block (256 in the Paper)
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualBlock(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)  # Add Relu at end as in ResNet v1 implementation
        return x
