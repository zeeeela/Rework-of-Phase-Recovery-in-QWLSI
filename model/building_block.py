import torch
import torch.nn as nn
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
I am taking reference (with modifications for my own learning) from:
1. https://debuggercafe.com/unet-from-scratch-using-pytorch/
2. https://github.com/sdsubhajitdas/Brain-Tumor-Segmentation/blob/master/bts/model.py
'''
def conv_double(filters, i, kernel_size_conv, padding_conv):
   '''
   Padded version
   Conv2d + ReLU + Conv2d 
   filters_down: list of filters for downsampling path including in_channels
   i: index of current layer [1, len(filters_down)]
   '''
   if len(filters_down) < 6:
       raise ValueError("filter down list must have at least 6 elements (including in_channels), currently have {len(filters_down)}")
   out = nn.Sequential(
        nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=filters[i], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv),
   )
   return out



class UNet_vanilla(nn.Module):
   def __init__(self, filters_down, kernel_size_conv, padding_conv):
       super(UNet_vanilla, self).__init__()
       self.conv1 = conv_double(filters=filters_down, i=1, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.conv2 = conv_double(filters=filters_down, i=2, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.conv3 = conv_double(filters=filters_down, i=3, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.conv4 = conv_double(filters=filters_down, i=4, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.conv5 = conv_double(filters=filters_down, i=5, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)

       self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

       self.uptrans1 = nn.ConvTranspose2d(in_channels=filters_up[0], out_channels=filters_up[1], kernel_size=kernel_transpose, stride=stride_transpose)
       self.up1 = conv_double(filters=filters_up, i=1, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.uptrans2 = nn.ConvTranspose2d(in_channels=filters_up[1], out_channels=filters_up[2], kernel_size=kernel_transpose, stride=stride_transpose)
       self.up2 = conv_double(filters=filters_up, i=2, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.uptrans3 = nn.ConvTranspose2d(in_channels=filters_up[2], out_channels=filters_up[3], kernel_size=kernel_transpose, stride=stride_transpose)
       self.up3 = conv_double(filters=filters_up, i=3, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.uptrans4 = nn.ConvTranspose2d(in_channels=filters_up[3], out_channels=filters_up[4], kernel_size=kernel_transpose, stride=stride_transpose)
       self.up4 = conv_double(filters=filters_up, i=4, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
       self.out = nn.Conv2d(in_channels=filters_up[4], out_channels=filters_up[5], kernel_size=1)


   def forward(self, x):
       x1 = self.conv1(x)
       p1 = self.pool(x1)
       x2 = self.conv2(p1)
       p2 = self.pool(x2)
       x3 = self.conv3(p2)
       p3 = self.pool(x3)
       x4 = self.conv4(p3)
       p4 = self.pool(x4)
       x5 = self.conv5(p4)

       x6 = self.uptrans1(x5)
       x7 = self.up1(torch.cat([x6, x4], dim=1))
       x8 = self.uptrans2(x7)
       x9 = self.up2(torch.cat([x8, x3], dim=1))
       x10 = self.uptrans3(x9)
       x11 = self.up3(torch.cat([x10, x2], dim=1))
       x12 = self.uptrans4(x11)
       x13 = self.up4(torch.cat([x12, x1], dim=1))
       x14 = self.out(x13)
       return x14


if __name__ == "__main__":
    model = UNet_vanilla(filters_down=filters_down, kernel_size_conv=kernel_size_conv, padding_conv=padding_conv)
    print(model)
    x = torch.randn((1, 1, 256, 256))  #batch_size=1, in_channels=1, H=256, W=256
    print("Input shape:", x.shape)
    y = model(x)
    print("Output shape:", y.shape)
