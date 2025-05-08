import torch
import torch.nn as nn
from audio_tokenizer.vqvae.models.residual import ResidualStack

class VQEncoder(nn.Module):
    """Simple CNN using 3 ConvLayer consisting
    of Conv2D, batch and Relu and a final Conv2d layer
    """

    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256], embedding_dim=64):
        """
        Parameters:
            in_channels (int) :
            hidden_dims (list[int]) :
            embedding_dim (int) : Size of output (i.e embeddings)
        """
        super().__init__()

        layers = []
        current_channels = in_channels
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = h_dim

        layers.append(nn.Conv2d(current_channels, embedding_dim, kernel_size=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)

'''
6 strided convolutions with stride 2 and window-size 
=> 64 smaller
'''
class RawEncoder(nn.Module):
    def __init__(self,in_channels = 1,hidden_dim=128,num_layers =6):
        super().__init__()
        layers = []
        for i in range(num_layers):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride = 2,
                    padding = 1,
                ),
                nn.ReLU()
            )
            layers.append(block)
        
        
        # => [B,hidden_dim,T//64]
        layers.append(
            ResidualStack(
                in_dim= hidden_dim,
                h_dim= hidden_dim,
                res_h_dim= hidden_dim,
                n_res_layers= 2,
            )
        )
        self.encoder = nn.Sequential(*layers)
        
    def forward(self,x):
        #[B,1,T] => [B,hidden_dim,T//64] = [B,dim embeddings,#of token]
        return self.encoder(x)
    
    
if __name__ == "__main__":
    x = torch.randint(low=-1,high=1,size = (16,1,32000),dtype=torch.float32)
    print(f"input shape:{x.shape}")
    model = RawEncoder()
    z_e = model(x)
    print(f"Output Shape : {z_e.shape}")