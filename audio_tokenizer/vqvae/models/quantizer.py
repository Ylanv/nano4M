import torch
import torch.nn as nn
import torch.nn.functional as F

''' 
Vector Quantizer, code is inspired from Misha Laskin VQ-VAE pytorch implementation
Github Repository : https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
'''
class VQuantizer(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,commitment_cost=0.25):
        ''' Init the quantizer module
        
        Parameters:
            num_embeddings (int) : Number of token in the vocabulary
            embedding_dim (int) : Dimension of each embedding
            commitement_cost (float) : Balance how strongly the encoder is encouraged to commit to the nearest codebook vector
        '''
        super().__init__()
        self.embedding_dim = embedding_dim 
        self.num_embeddings = num_embeddings 
        self.commitement_cost = commitment_cost
        
        # Codebook use for VQ 
        self.codebook = nn.Embedding(num_embeddings,embedding_dim)
        # Initialize weight uniformily between [-1/K,1/K] , K = voc_size
        self.codebook.weight.data.uniform_(-1.0/num_embeddings,1.0/num_embeddings)
        
    def forward(self,z):
            ''' Map z to a discrete one-hot vector that is index of the closest embeddings
            Parameters: 
                z (Tensor) : Output of the encoder module 
            '''
            
            # Reshape -> [B,H,W,C] and flatten 
            z = z.permute(0, 2, 3, 1).contiguous()
            z_flat = z.view(-1,self.embedding_dim) 
            
            # Calculate L2 distance between z and embeddings in the codebook
            z2 = torch.sum(z_flat**2,dim=1,keepdim=True) 
            e2 = torch.sum(torch.pow(self.codebook.weight,2),dim=1)
            ez = torch.matmul(z_flat,self.codebook.weight.t())
            distances = z2 + e2 -2*ez
            
            # Find closest            
            e_indices = torch.argmin(distances,dim=1)
            z_q = self.codebook(e_indices) 
            
            # Reshape back to [B, D, H, W]
            B, D, H, W = z.shape
            print(z.shape)
            z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            
            # Calculate loss 
            commitment_loss = F.mse_loss(z.detach(),z_q)
            codebook_loss = F.mse_loss(z,z_q.detach())
            loss = codebook_loss + self.commitement_cost * commitment_loss
            
            # Preserve gradients
            z_q = z + (z_q -z).detach()
            
            return z_q,loss,e_indices