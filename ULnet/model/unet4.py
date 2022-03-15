import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


class UNET4(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(UNET4, self).__init__()

        mid_ch = 16
        self.rebnconv1 = REBNCONV(in_ch,mid_ch,dirate=1)       #16
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)      #16
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch*2,dirate=1)     # 32
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch*2,mid_ch*2,dirate=2)   # 32

        self.rebnconv4d = REBNCONV(mid_ch*4, mid_ch*2, dirate=1)   #32
        self.rebnconv3d = REBNCONV(mid_ch*4,mid_ch*2,dirate=1)     #32
        self.rebnconv2d = REBNCONV(mid_ch*3,mid_ch,dirate=1)      #16
        self.rebnconv1d = REBNCONV(mid_ch*2,mid_ch,dirate=1)       #16

        self.side1 = nn.Conv2d(16,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(16,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(32,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(32,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(4,out_ch,1)

    def forward(self,x):

        hx = x

        hx1 = self.rebnconv1(hx)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        #decoder
        hx4d = self.rebnconv4d(torch.cat((hx, hx4), 1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4)
        d4 = _upsample_like(d4,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4)