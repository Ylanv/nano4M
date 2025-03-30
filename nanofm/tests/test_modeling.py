import torch
import pytest
from nanofm.modeling.transformer_layers import Mlp,Attention,Block,TransformerTrunk


# MLP Test
def test_mlp_forward_pass():
    # Test if the forward pass works with a simple input
    mlp = Mlp(in_features=64, hidden_features=128, out_features=32, bias=True)
    x = torch.rand(10, 64)  # Batch size = 10, in_features = 64
    output = mlp(x)
    
    # Check if output has the expected shape
    assert output.shape == (10, 32), f"Expected output shape (10, 32), but got {output.shape}"
    
# MLP Test
def test_mlp_forward_pass_3D():
    # Test if the forward pass works with a simple input
    mlp = Mlp(in_features=64, hidden_features=128, out_features=32, bias=True)
    x = torch.rand(1,10, 64)  # Batch size = 10, in_features = 64
    output = mlp(x)
    
    # Check if output has the expected shape
    assert output.shape == (1,10, 32), f"Expected output shape (10, 32), but got {output.shape}"


def test_mlp_with_default_out_features():
    # Test when out_features is not specified, it should default to in_features
    mlp = Mlp(in_features=64, hidden_features=128, bias=True)
    x = torch.rand(10, 64)  # Batch size = 10, in_features = 64
    output = mlp(x)
    
     # Check if output shape matches input shape when out_features is not provided
    assert output.shape == (10, 64), f"Expected output shape (10, 64), but got {output.shape}"
    
    
# Attention Test
def test_attention_forward():
    # Test basic forward pass
    B, L, D = 2, 10, 64  # Batch size, sequence length, dimension
    x = torch.rand(B, L, D)  # Random input tensor
    
    # Initialize the Attention layer
    attention_layer = Attention(dim=D, head_dim=16, qkv_bias=True, proj_bias=True)
    
    # Forward pass
    output = attention_layer(x)
    
    # Assert output shape: [B, L, D] (same as input)
    assert output.shape == (B, L, D), f"Expected output shape (2, 10, 64), but got {output.shape}"
    
    # Create a random tensor of shape [B, L, L] with values between 0 and 1
    mask_tensor = torch.rand(B, L, L)

    # Apply condition to create the boolean mask
    mask = mask_tensor >= 0.5 
    # Forward pass with mask
    output_with_mask = attention_layer(x, mask=mask)
    print("Output with mask:")
    print(output_with_mask.shape)  # Expected shape: [B, L, D]
    assert output_with_mask.shape == (B, L, D), f"Expected output shape (2, 10, 64), but got {output_with_mask.shape}"
    
def test_attention_mask_forward():
     # Test basic forward pass
    B, L, D = 2, 10, 64  # Batch size, sequence length, dimension
    x = torch.rand(B, L, D)  # Random input tensor
    
    # Initialize the Attention layer
    attention_layer = Attention(dim=D, head_dim=16, qkv_bias=True, proj_bias=False)
    
    # Create a random tensor of shape [B, L, L] with values between 0 and 1
    mask_tensor = torch.rand(B, L, L)

    # Apply condition to create the boolean mask
    mask = mask_tensor >= 0.5 
    # Forward pass with mask
    output_with_mask = attention_layer(x, mask=mask)
    assert output_with_mask.shape == (B, L, D), f"Expected output shape (2, 10, 64), but got {output_with_mask.shape}"

def test_block_forward():
    # Define the dimensions
    B, L, D = 2, 4, 8  # Batch size, sequence length, dimension
    head_dim = 2  # Dimension of each attention head (D // num_heads)
    mlp_ratio = 4  # MLP ratio
    use_bias = False  # Use bias in the layers

    # Create a random input tensor
    x = torch.rand(B, L, D)  # Random input tensor of shape [B, L, D]

    # Initialize the transformer block
    block = Block(dim=D, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias)

    # Forward pass without mask
    output = block(x)
    
    # Assert the output shape is correct [B, L, D]
    assert output.shape == (B, L, D), f"Expected output shape (2, 4, 8), but got {output.shape}"

def test_transformer_trunk_forward():
    # Define the dimensions
    B, L, D = 2, 4, 8  # Batch size, sequence length, dimension
    depth = 4  # Number of Transformer layers (blocks)
    head_dim = 2  # Dimension of each attention head (D // num_heads)
    mlp_ratio = 4  # MLP ratio
    use_bias = True  # Use bias in the layers

    # Create a random input tensor
    x = torch.rand(B, L, D)  # Random input tensor of shape [B, L, D]

    # Initialize the Transformer trunk
    transformer_trunk = TransformerTrunk(dim=D, depth=depth, head_dim=head_dim, mlp_ratio=mlp_ratio, use_bias=use_bias)

    # Forward pass without mask
    output = transformer_trunk(x)
    
    # Assert the output shape is correct [B, L, D]
    assert output.shape == (B, L, D), f"Expected output shape (2, 4, 8), but got {output.shape}"

  
