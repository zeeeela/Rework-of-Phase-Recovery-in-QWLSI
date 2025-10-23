import torch
import torch.nn as nn
from model.building_block import *

class unet_vanilla(nn.Module):
    def __init__(self, filters_down, filters_up, kernel_size_conv, padding_conv, pool_size, kernel_transpose, stride_transpose, depth):
        super(unet_vanilla, self).__init__()
        self.encoder = encoder_block(filters_down, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, pool_size=pool_size, depth=depth)
        self.decoder = decoder_block(filters_up, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, kernel_transpose=kernel_transpose, stride_transpose=stride_transpose, depth=depth)
        self.out = nn.Conv2d(in_channels=filters_up[4], out_channels=filters_up[5], kernel_size=1)

    def forward(self, x):
        enc_features = self.encoder(x)
        decoded = self.decoder(enc_features)
        decoded = self.out(decoded) 
        return decoded


class unet_block_att(nn.Module):
    def __init__(self, filters_down, filters_up, kernel_size_conv, padding_conv, pool_size, kernel_transpose, stride_transpose, depth):
        super(unet_block_att, self).__init__()
        self.encoder = encoder_block(filters_down, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, pool_size=pool_size, depth=depth)
        self.decoder = decoder_block_att(filters_up, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, kernel_transpose=kernel_transpose, stride_transpose=stride_transpose, depth=depth, attn=True)
        self.out = nn.Conv2d(in_channels=filters_up[4], out_channels=filters_up[5], kernel_size=1)

    def forward(self, x):
        enc_features = self.encoder(x)
        decoded = self.decoder(enc_features)
        decoded = self.out(decoded) 
        return decoded





if __name__ == "__main__":
    from ml_collections import ConfigDict
    from config.config import get_config

    config = get_config()
    filters_down = config.model.filters_down
    filters_up = config.model.filters_up
    kernel_size_conv = config.model.kernel_size_conv
    padding_conv = config.model.padding_conv
    pool_size = config.model.pool_size
    kernel_transpose = config.model.kernel_size_conv_transpose
    stride_transpose = config.model.stride_transpose

    '''
    select model type: vanilla UNet or attention UNet
    we will check both implementations through their output shapes
    '''

    
    model = input("Enter model type (vanilla/attention): ")
    if model == "vanilla":
        model = unet_vanilla(filters_down=filters_down, filters_up=filters_up, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, pool_size=pool_size, kernel_transpose=kernel_transpose, stride_transpose=stride_transpose, depth=5)
    elif model == "attention":
        model = unet_block_att(filters_down=filters_down, filters_up=filters_up, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv, pool_size=pool_size, kernel_transpose=kernel_transpose, stride_transpose=stride_transpose, depth=5)
    else:
        raise ValueError("Invalid model type. Choose 'vanilla' or 'attention'.")

    print(model)
    x = torch.randn((1, 1, 256, 256))
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)

