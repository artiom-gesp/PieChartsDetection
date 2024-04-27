import torch
import torch.nn as nn

from ._cpools import TopPool, BottomPool, LeftPool, RightPool

class ConvolutionalBlock(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(ConvolutionalBlock, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu    

class CrossPoolingLayer(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(CrossPoolingLayer, self).__init__()
        self.p1_conv1 = ConvolutionalBlock(3, dim, 128)
        self.p2_conv1 = ConvolutionalBlock(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvolutionalBlock(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)
        pool1 = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)
        pool2 = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class CenterPoolingLayer(CrossPoolingLayer):
    def __init__(self, dim):
        super(CenterPoolingLayer, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)


def make_ct_layer(dim):
    return CenterPoolingLayer(dim)
