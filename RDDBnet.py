import functools
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


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
        global pool_1,pool_0
        pool_1 = x
        x += pool_0

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)  # C128, 256
        global pool_2,pool_9
        pool_2 = x
        x += pool_9

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)  # C128, 128,128
        global pool_3, pool_8
        pool_3 = x
        x += pool_8

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)  # C256, 64,64

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)  # C128, 64,64
        global pool_4, pool_7
        pool_4 = x
        x += pool_7


        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)  # C64, 64,64
        global pool_5, pool_6
        pool_5 = x
        x += pool_6


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
        global pool_5, pool_6
        pool_6 = x
        x += pool_5

        x = self.decon2(x)
        x = self.bn2(x)
        x = self.relu2(x)  # C128, 64
        global pool_4, pool_7
        pool_7 = x
        x += pool_4

        x = self.decon3(x)
        x = self.bn3(x)
        x = self.relu3(x)  # C128, 128,128
        global pool_3, pool_8
        pool_8 = x
        x += pool_3

        x = self.decon4(x)
        x = self.bn4(x)
        x = self.relu4(x)  # C256, 256

        x = self.decon5(x)
        x = self.bn5(x)
        x = self.relu5(x)  # C128, 256
        global pool_2, pool_9
        pool_9 = x
        x += pool_2

        x = self.decon6(x)
        x = self.bn6(x)
        x = self.relu6(x)  # C64, 256
        global pool_1, pool_0
        pool_0 = x
        x += pool_1
        return x




class RDDBNetA(nn.Module):
    def __init__(self,in_nc, out_nc, nf, nb=2, gc = 32):
        super(RDDBNetA, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3,1,1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf,3,1,1, bias=True)
        self.decode = Decoder()
        self.conv_last = nn.Conv2d(nf, out_nc,3,1,1,bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.decode(fea)
        out = self.conv_last(fea)

        return out

class RDDBNetB(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb=2, gc=32):
        super(RDDBNetB, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.encode = Encoder()
        self.conv_last = nn.Conv2d(nf, out_nc,3,1,1,bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.encode(fea)
        out = self.conv_last(fea)

        return out


if __name__ == "__main__":
    # Hyper Parameters
    x = torch.FloatTensor(np.random.random((1, 3, 256, 256)))
    y = torch.FloatTensor(np.random.random((1, 1, 64, 64)))
    # init pool
    pool_1, pool_2, pool_3, pool_4, pool_5 = np.zeros(5)
    pool_6, pool_7, pool_8, pool_9, pool_0 = np.zeros(5)
    generatorA = RDDBNetA(3,1,64, nb=1)
    generatorB = RDDBNetB(1,3,64, nb=1)
    D = Discriminator(y)

    genB = generatorA(x)
    rectB = generatorB(genB)
    # 还原 pool
    # pool_1, pool_2, pool_3, pool_4, pool_5 = np.zeros(5)
    # pool_6, pool_7, pool_8, pool_9, pool_0 = np.zeros(5)

    genA = generatorB(y)
    rectA = generatorA(genA)


    # print("test>:")
    # print(" Network output ", gen_y)
    print(generatorA)
    print(generatorB)
    #print(gobal(pool_1))
    print(pool_5.shape)
    print("# #"*20)
    print(pool_0.shape)
    #print(gen_y)
    # print(type(x))
