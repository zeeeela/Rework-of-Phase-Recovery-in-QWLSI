import torch
import torch.nn as nn


'''
I am taking reference (with modifications for my own learning) from:

1. https://github.com/sdsubhajitdas/Brain-Tumor-Segmentation/blob/master/bts/model.py
2. https://gist.github.com/shuuchen/6d39225b018d30ccc86ef10a4042b1aa
3. https://medium.com/@AIchemizt/attention-u-net-in-pytorch-step-by-step-guide-with-code-and-explanation-417d80a6dfd0
4. 
'''

####### Universal Building Blocks for UNet #######
class conv_double(nn.Module):
    def __init__(self, filters, i, kernel_size_conv, padding_conv):
        super(conv_double, self).__init__()
        if len(filters) < 6:
            raise ValueError(f"filter list must have at least 6 elements, currently have {len(filters)}")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(num_features=filters[i]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[i], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(num_features=filters[i]),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class encoder_block(nn.Module):
    def __init__(self, filters, kernel_size_conv, padding_conv, pool_size, depth=5):
        super(encoder_block, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.conv = nn.ModuleList()
        for i in range(1, depth+1):
            self.conv.append(conv_double(filters, i, kernel_size_conv, padding_conv))
   
        
    def forward(self, x):
        out = []
        for i, conv_layer in enumerate(self.conv):
            if i>0:
                x = self.pool(x)
            x = conv_layer(x)
            out.append(x)
        return out

class deconv_double(nn.Module):
    def __init__(self, filters, i, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose):
        super(deconv_double, self).__init__()

        self.uptrans = nn.ConvTranspose2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=kernel_transpose, stride=stride_transpose)
        self.up = conv_double(filters, i, kernel_size_conv, padding_conv)

    def forward(self, x, enc):
        x = self.uptrans(x)
        x = torch.cat([x, enc], dim=1)
        x = self.up(x)
        return x



class decoder_block(nn.Module):
    def __init__(self, filters, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose, depth=5):
        super(decoder_block, self).__init__()

        self.depth = depth
        self.deconv = nn.ModuleList()
        for i in range(1, self.depth):
            self.deconv.append(deconv_double(filters, i, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose))

    def forward(self, enc_features):
        for i, deconv_layer in enumerate(self.deconv):
            if i==0:
                x = enc_features.pop()
            bridge = enc_features.pop()
            x = deconv_layer(x, bridge)

        return x
    
######### Attention Building Blocks for UNet #########

class att_gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(att_gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class deconv_double_att(nn.Module):
    def __init__(self, filters, i, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose, attn=None):
        super(deconv_double_att, self).__init__()

        self.uptrans = nn.ConvTranspose2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=kernel_transpose, stride=stride_transpose)
        self.up = conv_double(filters, i, kernel_size_conv, padding_conv)
        self.att = att_gate(F_g=filters[i], F_l=filters[i], F_int=filters[i]//2)
        self.attn = attn

    def forward(self, x, enc):
        x = self.uptrans(x)
        if self.attn:
            enc = self.att(g=x, x=enc)
        x = torch.cat([x, enc], dim=1)
        x = self.up(x)
        return x
    

class decoder_block_att(nn.Module):
    def __init__(self, filters, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose, depth=5, attn=True):
        super(decoder_block_att, self).__init__()

        self.depth = depth
        self.deconv = nn.ModuleList()
        for i in range(1, self.depth):
            if attn:
                self.deconv.append(deconv_double_att(filters, i, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose, attn=True))
            else:
                self.deconv.append(deconv_double_att(filters, i, kernel_size_conv, padding_conv, kernel_transpose, stride_transpose, attn=False))       

    def forward(self, enc_features):
        for i, deconv_layer in enumerate(self.deconv):
            if i==0:
                x = enc_features.pop()
            bridge = enc_features.pop()
            x = deconv_layer(x, bridge)

        return x


######### Residual Building Blocks for UNet #########
class ResidualBlock(nn.Module):
    def __init__(self, filters, i, kernel_size_conv, padding_conv):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=filters[i-1], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv)
        self.bn1 = nn.BatchNorm2d(num_features=filters[i-1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=filters[i], out_channels=filters[i], kernel_size=kernel_size_conv, padding=padding_conv)
        self.bn2 = nn.BatchNorm2d(num_features=filters[i])
    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


