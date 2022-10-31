import torchvision
import torch
import torch.nn as nn

class OneConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, height, width): # Padding mode = zeros - default
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels)) 

    def forward(self, x):
        x = self.double_conv(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, height, width): # Padding mode = zeros - default
        super().__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='zeros', bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels))  # ,nn.Dropout())

        # self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.LayerNorm([out_channels, height, width]),
        #                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.LayerNorm([out_channels, height, width]),
        #                                  nn.BatchNorm2d(out_channels)) #,nn.Dropout())

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DoubleConv_up(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, height, width): # Padding mode = zeros - default
        super().__init__()
        self.double_conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,stride=2, padding=1, padding_mode='zeros', bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,stride=2, padding=1, padding_mode='zeros', bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels))  

    def forward(self, x):
        x = self.double_conv(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, height, width))
        self.maxpool_conv = nn.Sequential(nn.AvgPool2d(2), DoubleConv(in_channels, out_channels, height, width))

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, height, width)

        # ako radim sum onda mora ovako 
        self.conv_jedan = OneConv(in_channels, out_channels, height, width)
        # self.conv_up = DoubleConv_up(in_channels, out_channels, height, width)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        # # x2 = self.crop(x2, x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)  


        # x1 = self.up(x1)
        # x1 = self.conv_jedan(x1)
        # x = torch.add(x1,x2)
        # x = self.conv(x)          
        # # # x1 = self.conv_up(x1)            
        # # # x = torch.add(x1,x2)
        
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        return enc_ftrs


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_conv = self.conv(x)
        # x_prob = self.softmax(x_conv)

        return x_conv


class UNet16_4(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet16_4, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 64, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(64, 256, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(256, 1024, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(1280, 256, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(320, 64, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(80, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)
        # self.weight_init()

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x


class UNet16_5(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet16_5, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 80, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(80, 400, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(400, 2000, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(2400, 400, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(480, 80, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(96, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)
        # self.weight_init()

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x


class UNet16_2(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet16_2, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 32, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(32, 64, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(64, 128, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(192, 64, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(96, 32, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(48, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)
        # self.weight_init()

        # ################### KAD JE SUM A NE CONCAT
        # self.inc = DoubleConv(n_channels, 16, height, width)
        # self.down1 = Down(16, 32, int(height / 2), int(width / 2))  # 256x
        # self.down2 = Down(32, 64, int(height / 4), int(width / 4))  # 128x
        # self.down3 = Down(64, 128, int(height / 8), int(width / 8))  # 64x
        # self.up1 = Up(128, 64, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(64, 32, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(32, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16, n_classes)
        # # self.weight_init()


    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x


### 16 REZOLUCIJA
class UNet16_3(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet16_3, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 48, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(48, 144, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(144, 432, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(576, 144, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(192, 48, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(64, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    # normal_init(m)
                    torch.nn.init.normal_(m, mean=0.0, std=1.0)
            except:
                # normal_init(block)
                torch.nn.init.normal_(block, mean=0.0, std=1.0)


### 16 REZOLUCIJA
class UNet16_3_4(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet16_3_4, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        # self.inc = DoubleConv(n_channels, 16, height, width)
        # self.down1 = Down(16, 48, int(height / 2), int(width / 2))  # 256x
        # self.down2 = Down(48, 144, int(height / 4), int(width / 4))  # 128x
        # self.down3 = Down(144,432 , int(height / 8), int(width / 8))  # 64x
        # self.up1 = Up(576,144, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(192, 48, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(64, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16, n_classes)

        self.inc = DoubleConv(n_channels, 16, height, width)
        self.down1 = Down(16, 48, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(48, 144, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(144, 432, int(height / 8), int(width / 8))  # 64x
        self.down4 = Down(432, 1296, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(1728, 432, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(576, 144, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up3 = Up(192, 48, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up4 = Up(64, 16, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    # normal_init(m)
                    torch.nn.init.normal_(m, mean=0.0, std=1.0)
            except:
                # normal_init(block)
                torch.nn.init.normal_(block, mean=0.0, std=1.0)


### 32 REZOLUCIJA
class UNet32(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet32, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)
        # povecan broj ulaznih kanala sa 16 na 32
        # self.inc = DoubleConv(n_channels, 16*2, height, width)
        # self.down1 = Down(16*2, 24*2, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(24*2, 32*2, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(32*2, 48*2, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(80*2, 32*2, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(56*2, 24*2, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(40*2, 16*2, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16*2, n_classes)
        ############################################
        # self.inc = DoubleConv(n_channels, 16, height, width)
        # self.down1 = Down(16, 32, int(height / 2), int(width / 2)) # 256x
        # self.down2 = Down(32, 48, int(height / 4), int(width / 4)) # 128x
        # self.down3 = Down(48, 64, int(height / 8), int(width / 8)) # 64x
        # self.up1 = Up(112, 48, int(height / 4), int(width / 4)) # 128x, in_channels = 32 + 64, out_channels = 32
        # self.up2 = Up(80, 32, int(height / 2), int(width / 2)) # 256x, in_channels = 16 + 32, out_channels = 16
        # self.up3 = Up(48, 16, int(height), int(width)) # 512x, in_channels = 8 + 16, out_channels = 8
        # self.outc = OutConv(16, n_classes)

        #############################################
        #
        self.inc = DoubleConv(n_channels, 32, height, width)
        self.down1 = Down(32, 64, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(64, 128, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(128, 256, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(384, 128, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(192, 64, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(96, 32, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(32, n_classes)

        # self.weight_init()

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    # normal_init(m)
                    torch.nn.init.normal_(m, mean=0.0, std=1.0)
            except:
                # normal_init(block)
                torch.nn.init.normal_(block, mean=0.0, std=1.0)


### 32 REZOLUCIJA
class UNet0(nn.Module):
    def __init__(self, n_channels, n_classes, height, width, zscore):
        super(UNet0, self).__init__()
        self.zscore = zscore
        self.normalization = nn.InstanceNorm2d(n_channels, affine=True)

        self.inc = DoubleConv(n_channels, 64, height, width)
        self.down1 = Down(64, 128, int(height / 2), int(width / 2))  # 256x
        self.down2 = Down(128, 256, int(height / 4), int(width / 4))  # 128x
        self.down3 = Down(256, 512, int(height / 8), int(width / 8))  # 64x
        self.up1 = Up(768, 256, int(height / 4), int(width / 4))  # 128x, in_channels = 32 + 64, out_channels = 32
        self.up2 = Up(384, 128, int(height / 2), int(width / 2))  # 256x, in_channels = 16 + 32, out_channels = 16
        self.up3 = Up(192, 64, int(height), int(width))  # 512x, in_channels = 8 + 16, out_channels = 8
        self.outc = OutConv(64, n_classes)

        # self.weight_init()

    def forward(self, x):
        if self.zscore == 0:
            x = self.normalization(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    # normal_init(m)
                    torch.nn.init.normal_(m, mean=0.0, std=1.0)
            except:
                # normal_init(block)
                torch.nn.init.normal_(block, mean=0.0, std=1.0)

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.xavier_uniform(m.weight, gain=numpy.sqrt(2.0))
#         init.constant(m.bias, 0.1)
