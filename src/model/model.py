import functools
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def get_deconv_params(upscale_factor):
    """
    args:
        ratio:(str) upsample level
        H_out =(H_in−1)×stride[0]−2×pad[0]+ksize[0]+output_padding[0] 
    """
    if upscale_factor==2:
        kernel_size = 2
        stride=2
    elif upscale_factor==4:
        kernel_size = 2
        stride=4
    elif upscale_factor==8:
        kernel_size = 4
        stride=8
    output_padding = (stride-kernel_size)
    return kernel_size, stride, output_padding


def deconv(in_planes, out_planes, upscale_factor=2):
    """2d deconv"""
    kernel_size, stride, opad = get_deconv_params(upscale_factor)
    # print("DECONV", kernel_size, stride, opad)
    return nn.ConvTranspose2d(in_planes, out_planes, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=0, 
                              bias=False,
                              output_padding=opad,
                              groups=1)


class ESPCN(nn.Module):
    """
    Superresolution using an efficient sub-pixel convolutional neural network
    Reference:
        CVPR2016: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
        code: https://github.com/pytorch/examples/tree/master/super_resolution
        PixelShuffle: https://blog.csdn.net/gdymind/article/details/82388068
    """

    def __init__(self,
                 nb_channel=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(ESPCN, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1, 1 / 2]]

        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            nb_channel, kernels[0], kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            kernels[0], kernels[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            kernels[1], kernels[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            kernels[2], base_kernel * upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv5 = nn.Conv2d(
            base_kernel, nb_channel, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return self.conv5(x)


class SRCNN(torch.nn.Module):
    """
    Superresolution using a Deep Convolutional Network
    Reference:
        ECCV2014: Learning a Deep Convolutional Network for Image Super-Resolution
    """

    def __init__(self,
                 nb_channel=3,
                 upscale_factor=2,
                 base_kernel=64):
        super(SRCNN, self).__init__()
        kernels = [int(x * base_kernel) for x in [1, 1 / 2]]
        self.up = upscale_factor
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(
            nb_channel, kernels[0], kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(
            kernels[0], kernels[1], kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(
            kernels[1], nb_channel, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.up, mode="nearest")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class EDSR(nn.Module):
    """
    https://github.com/icpm/super-resolution/edit/master/EDSR/model.py
    """
    def __init__(self, nb_channel, upscale_factor=2, base_channel=64, num_residuals=50):
        super(EDSR, self).__init__()

        self.input_conv = nn.Conv2d(nb_channel, base_channel, kernel_size=3, stride=1, padding=1)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_channel, kernel=3, stride=1, padding=1))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        upscale = []
        for _ in range(int(math.log2(upscale_factor))):
            upscale.append(deconv(base_channel, base_channel, upscale_factor=2))
        self.upscale_layers = nn.Sequential(*upscale)

        self.output_conv = nn.Conv2d(base_channel, nb_channel, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.residual_layers(x)
        x = self.mid_conv(x)
        x = torch.add(x, residual)
        x = self.upscale_layers(x)
        x = self.output_conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, num_channel, kernel=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel, stride, padding)
        self.gn = nn.GroupNorm(32, num_channel) # replace BN => GN
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.gn(self.conv1(x))
        x = self.activation(x)
        x = self.gn(self.conv2(x))
        x = torch.add(x, residual)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor, kernel=3, stride=1, padding=1):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel * upscale_factor ** 2, kernel, stride, padding)
        self.ps = nn.PixelShuffle(upscale_factor)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding)

    def forward(self, x):
        x = self.ps(self.conv1(x))
        return x


class ResidualDenseBlock_5(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5, self).__init__()
        # gc = growth channeal, image = 256
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)  # 64 -32,256
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)  # 96, 32, 256
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)  # 128, 32, 256
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)  # 160, 32, 256
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)  # 192, 32, 256
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x, lemda=0.2):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * lemda + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5(nf, gc)
        self.RDB2 = ResidualDenseBlock_5(nf, gc)
        self.RDB3 = ResidualDenseBlock_5(nf, gc)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, lemda=0.2):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * lemda + x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)  # C64 256

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)  # C128 256

        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)  # dwonsampling
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.1)  # C128 128

        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)  # downsampling
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.1)  # C256 64

        self.conv5 = nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.LeakyReLU(0.1) # C128 64

        self.conv6 = nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.LeakyReLU(0.1)  # C64 64

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) # C64 256 256

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)  # C128, 256

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)  # C128, 128,128

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)  # C256, 64,64

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)  # C128, 64,64

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)  # C64, 64,64


        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.decon1 = nn.ConvTranspose2d(64,64,3,stride=1,padding=1, output_padding=0,bias=False)
        self.bn1 =nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()  # C64 64

        self.decon2 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1, output_padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()  # C128 64

        self.decon3 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()  # C128 128

        self.decon4 = nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()  # C256 256

        self.decon5 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1, output_padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()  # C128 256

        self.decon6 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1, output_padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()  # C64 256

    def forward(self,x):
        x = self.decon1(x)
        x = self.bn1(x)
        x = self.relu1(x)  # C64 64

        x = self.decon2(x)
        x = self.bn2(x)
        x = self.relu2(x)  # C128, 64

        x = self.decon3(x)
        x = self.bn3(x)
        x = self.relu3(x)  # C128, 128,128

        x = self.decon4(x)
        x = self.bn4(x)
        x = self.relu4(x)  # C256, 256

        x = self.decon5(x)
        x = self.bn5(x)
        x = self.relu5(x)  # C128, 256

        x = self.decon6(x)
        x = self.bn6(x)
        x = self.relu6(x)  # C64, 256

        return x



class RDDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc = 32, mode='x2'):
        super(RDDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3,1,1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf,3,1,1, bias=True)
        self.upconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mode = mode
        self.nb = nb
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     # fea = self.conv_first(x)
    #     # trunk = self.trunk_conv(self.RRDB_trunk(fea))
    #     # fea = fea + trunk
    #     # fea = self.decode(fea)
    #     #
    #     # out = self.conv_last(fea)
    #     #
    #     # return out
    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        if self.mode == 'x4':
            fea = self.lrelu(self.upconv(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.mode == 'x2':
            fea = self.lrelu(self.upconv(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.mode == 'x1':
            fea = self.lrelu(self.upconv(fea))

        fea = self.lrelu(self.HRconv(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out



class RDDBNetB(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb=3, gc=32, mode='x2'):
        super(RDDBNetB, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.mode = mode
        self.nb = nb
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        # fea = self.encode(fea)
        # out = self.conv_last(fea)

        if self.mode == 'x4':
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.mode == 'x2':
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv1(fea))
        fea = self.lrelu(self.HRconv(fea))
        fea = self.lrelu(self.HRconv(fea))
        fea = self.lrelu(self.HRconv(fea))
        fea = self.lrelu(self.HRconv(fea))
        fea = self.lrelu(self.HRconv(fea))
        fea = self.lrelu(self.HRconv(fea))

        fea = self.lrelu(self.HRconv(fea))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# class Discriminator(nn.Module):
#     def __init__(self, in_nc, nf = 64):
#         super(Discriminator, self).__init__()
#         #[inchan -- 64,64,64 / 256]
#         self.conv0 = nn.Conv2d(in_nc, nf, 3,1,1,bias=True)
#         self.Lrelu0 = nn.LeakyReLU(0.2)
#
#         #[64--128.32,32 /128]
#         self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
#         self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
#         self.Lrelu1_0 = nn.LeakyReLU(0.2)
#         self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 3, 2, 1, bias=False)
#         self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
#         self.Lrelu1_1 = nn.LeakyReLU(0.2)
#
#         #[128--256,16,16 / 64]
#         self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
#         self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
#         self.Lrelu2_0 = nn.LeakyReLU(0.2)
#         self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=False)
#         self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
#         self.Lrelu2_1 = nn.LeakyReLU(0.2)
#
#         #[256--512.8,8/32]
#         self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
#         self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.Lrelu3_0 = nn.LeakyReLU(0.2)
#         self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 3, 2, 1, bias=False)
#         self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.Lrelu3_1 = nn.LeakyReLU(0.2)
#
#        ### 需要把分辨率学成一样吗？
#         # dens layer ????
#         self.dense = nn.Linear(512,1024)
#         self.LreluD =nn.LeakyReLU(0.2)
#         self.line = nn.Linear(1024,1)
#         self.lastrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self,x):
#         # 输入
#         x = self.conv0(x)
#         x = self.Lrelu0(x)
#
#         # con1
#         x = self.conv1_0(x)
#         x = self.bn1_0(x)
#         x = self.Lrelu1_0(x)
#         x = self.conv1_1(x)
#         x = self.bn1_1(x)
#         x = self.Lrelu1_1(x)
#
#         # con2
#         x = self.conv2_0(x)
#         x = self.bn2_0(x)
#         x = self.Lrelu2_0(x)
#         x = self.conv2_1(x)
#         x = self.bn2_1(x)
#         x = self. Lrelu2_1(x)
#
#         # con3
#         x = self.conv3_0(x)
#         x = self.bn3_0(x)
#         x = self.Lrelu3_0(x)
#         x = self.conv3_1(x)
#         x = self.bn3_1(x)
#         x = self.Lrelu3_1(x)
#         print(x.shape)
#
#         # Linear
#         x = self.dense(x)
#         x = self.LreluD(x)
#         x = self.line(x)
#         x = self.lastrelu(x)
#         out = self.sigmoid(x)
#
#         return out
# class D_LR(nn.Module):
#     def __init__(self, nc=1,ndf=64):
#         super(D_LR, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input):
#         return self.main(input)[:,0,0,0]
#
# class D_HR(nn.Module):
#     def __init__(self,nc=3,ndf=64):
#         super(D_HR, self).__init__()
#         self.main = nn.Sequential(
#             # input is (nc) x 256 x 256
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 128 x 128
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 64 x 64
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 32 x 32
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 16 x 16
#             nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),
#             # stat size .(ndf*16)x 8 x 8
#             nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input):
#         return self.main(input)[:,0,0,0]


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

#### SRDenseNet

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class SRDenseNetA(nn.Module):
    def __init__(self, in_nc, out_nc, nb_channel=1, growth_rate=16, num_blocks=8, num_layers=8, mode='x2'):
        super(SRDenseNetA, self).__init__()
        self.mode = mode
        self.conv_first = nn.Conv2d(in_nc, 1, 3, 1, 1, bias=True)


        # low level features
        self.conv = ConvLayer(nb_channel, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # deconvolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=3 // 2, output_padding=1),
            nn.ReLU(inplace=True),
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, nb_channel, kernel_size=3, padding=3 // 2)

        self.conv_last = nn.Conv2d(1, out_nc, 3, 1, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        if self.mode == 'x2':
            x = self.deconv(x)
        elif self.mode == 'x4':
            x = self.deconv(x)
            x = self.deconv(x)
        x = self.reconstruction(x)
        x = self.conv_last(x)
        return x


class SRDenseNetB(nn.Module):
    def __init__(self, in_nc, out_nc, nb_channel=1, growth_rate=16, num_blocks=8, num_layers=8, mode='x2'):
        super(SRDenseNetB, self).__init__()
        self.mode = mode
        self.conv_first = nn.Conv2d(in_nc, 1, 3, 1, 1, bias=True)


        # low level features
        self.conv = ConvLayer(nb_channel, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # deconvolution layers
        self.deconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, nb_channel, kernel_size=3, padding=3 // 2)

        self.conv_last = nn.Conv2d(1, out_nc, 3, 1, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv(x)
        x = self.dense_blocks(x)
        x = self.bottleneck(x)
        if self.mode == 'x2':
            x = self.deconv(x)
        elif self.mode == 'x4':
            x = self.deconv(x)
            x = self.deconv(x)
        x = self.reconstruction(x)
        x = self.conv_last(x)
        return x
if __name__ == "__main__":
    # Hyper Parameters
    x = torch.FloatTensor(np.random.random((1, 3, 128, 128)))
    y = torch.FloatTensor(np.random.random((1, 1, 64, 64)))
    x1 = F.max_pool2d(x, 3, 4, padding=0, dilation=1, ceil_mode=False, return_indices=False)
    x1 = F.conv2d(x1,torch.FloatTensor(1,3,1,1),bias=None, stride=1)

    print(y.shape)

    #generatorA = RDDBNetA(3,1,64,nb=1,mode='x2')
    #generatorB = RDDBNetB(1,3,64,nb=1,mode='x2')

    G_A  = SRDenseNetA(1,3)
    G_B = SRDenseNetB(3,1)


    genB = G_A(y)
    genA = G_B(x)
    print('genA：', genA.size())
    print('genB：', genB.size())
    #rectB = generatorB(genB)
    # 还原 pool
    # pool_1, pool_2, pool_3, pool_4, pool_5 = np.zeros(5)
    # pool_6, pool_7, pool_8, pool_9, pool_0 = np.zeros(5)

    genA = generatorB(y)
    rectA = generatorA(genA)
    print('genA：', genA.size())
    print('recA：', rectA.size())
    print('genB：', genB.size())
    print('recB：', rectB.size())

    netD_B = D_LR(nc=1)
    print(netD_B (y).shape)
    D = D_HR()
    print(D(x).shape)


    # print("test>:")
    # print(" Network output ", gen_y)
    # print(generatorA)
    # print(generatorB)
    #print(gobal(pool_1))
    #print(gen_y)
    # print(type(x))
