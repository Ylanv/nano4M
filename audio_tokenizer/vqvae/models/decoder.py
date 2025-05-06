import torch
import torch.nn as nn


class VQDecoder(nn.Module):
    """Decoder, mirror of the encoder, use ConvTranspose2d for upsampling"""

    def __init__(self, embedding_dim=64, hidden_dims=[256, 128, 64], out_channels=1):
        """Init the VQDecoder

        Parameters:
            embedding_dim (int) : Dimension of each embeddings
            hidden_dims (list[int]) : Dimensions of hidden layer
            out_channels (int) :
        """
        super().__init__()

        layers = []
        current_channels = embedding_dim
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels, h_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = h_dim

        # Final layer to get back to mel-spectrogram shape
        layers.append(
            nn.ConvTranspose2d(
                current_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,  # Keep output resolution
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
