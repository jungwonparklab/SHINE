import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class Basic_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(Basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class conv_lr(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv_lr, self).__init__()
        self.conv = Basic_conv(in_planes, out_planes, 3,1,1)
    
    def forward(self, x):
        return self.conv(x)

class conv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3,1,1),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)   

class Encode(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.Pconv1 = conv_lr(in_channels,out_channels)
        self.Pconv2 = conv_lr(out_channels,out_channels)
        self.maxpool = nn.MaxPool2d(scale)

    def forward(self,x):
        p1 = self.Pconv1(x)
        p2 = self.Pconv2(p1)
        out = self.maxpool(p2)
        return out

class Encode_lr(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.Pconv1 = conv_lr(in_channels,out_channels)
        self.maxpool = nn.MaxPool2d(scale)

    def forward(self,x):
        p1 = self.Pconv1(x)
        out = self.maxpool(p1)
        return out

class Encode_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.Pconv1 = conv_lr(in_channels,out_channels)
        self.maxpool = nn.MaxPool2d(scale)
        self.Pconv2 = conv_lr(out_channels,out_channels)
        self.Upsample = nn.Upsample(scale_factor=scale)


    def forward(self,x):
        p1 = self.Pconv1(x)
        out = self.maxpool(p1)
        out = self.Pconv2(out)
        return self.Upsample(out)

class Decode_lr(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.conv1 = conv_lr(in_channels,out_channels)
        self.conv2 = conv_lr(out_channels,out_channels)
        self.Upsample = nn.Upsample(scale_factor=scale)

    def forward(self,x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        out = self.Upsample(p2)
        return out

class Decode(nn.Module):
    def __init__(self, in_channels,middle_channels1,middle_channels2, out_channels):
        super().__init__()
        self.conv1 = conv_lr(in_channels,middle_channels1)
        self.conv2 = conv_lr(middle_channels1,middle_channels2)
        self.conv3 = conv(middle_channels2, out_channels)
        #self.params = nn.Parameter(self.conv3)
        #nn.init.zeros_(self.conv3.parameters)
        if isinstance(self.conv3, nn.Conv2d):
            torch.nn.init.zeros_(self.conv3.weight)
            self.conv3.bias.data.fill_(0.01)

    def forward(self,x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        out = self.conv3(p2)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.encode1 = Encode(in_channels,48)
        self.encode2 = Encode_lr(48,48)
        self.encode3 = Encode_lr(48,48)
        self.encode4 = Encode_lr(48,48)
        self.encode5 = Encode_bottleneck(48,48)
        self.decode1 = Decode_lr(96,96)
        self.decode2 = Decode_lr(144,96)
        self.decode3 = Decode_lr(144,96)
        self.decode4 = Decode_lr(144,96)
        self.decode5 = Decode(96+in_channels,64,32,out_channels)

    def forward(self,x, shuffle=0):
        N, C, nH, nW = x.shape
        if nH % 64 != 0 or nW % 64 != 0:
            x = F.pad(x, [0, 64 - nW % 64, 0, 64 - nH % 64], mode = 'constant')
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        concat4 = torch.cat((pool4,pool5),dim=1)
        upsample4 = self.decode1(concat4)
        concat4 = torch.cat((upsample4,pool3),dim=1)
        upsample3 = self.decode2(concat4)
        concat3 = torch.cat((upsample3,pool2),dim=1)
        upsample2 = self.decode3(concat3)
        concat2 = torch.cat((upsample2,pool1),dim=1)
        upsample1 = self.decode4(concat2)
        concat1 = torch.cat((upsample1,x),dim=1)
        out = self.decode5(concat1)
        if nH % 64 != 0 or nW % 64 != 0:
            out = out[:, :, 0:nH, 0:nW]
        return out

    def inference(self,x, shuffle=0):
        pool1 = self.encode1(x)
        pool2 = self.encode2(pool1)
        pool3 = self.encode3(pool2)
        pool4 = self.encode4(pool3)
        pool5 = self.encode5(pool4)
        concat4 = torch.cat((pool4,pool5),dim=1)
        upsample4 = self.decode1(concat4)
        concat4 = torch.cat((upsample4,pool3),dim=1)
        upsample3 = self.decode2(concat4)
        concat3 = torch.cat((upsample3,pool2),dim=1)
        upsample2 = self.decode3(concat3)
        concat2 = torch.cat((upsample2,pool1),dim=1)
        upsample1 = self.decode4(concat2)
        concat1 = torch.cat((upsample1,x),dim=1)
        out = self.decode5(concat1)
        return out
