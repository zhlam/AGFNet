"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
Dlinknet+DCAM+GAFF
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu,inplace=True)


#注意力机制
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

#扩张模块
class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.CoordAtt = CoordAtt(channel,channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        CoordAtt1_out = self.CoordAtt(dilate1_out)
        CoordAtt2_out = self.CoordAtt(dilate2_out)
        CoordAtt3_out = self.CoordAtt(dilate3_out)
        CoordAtt4_out = self.CoordAtt(dilate4_out)

        out = x + CoordAtt1_out + CoordAtt2_out + CoordAtt3_out + CoordAtt4_out
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

#GFFM
#多尺度条状卷积
class MSCM(nn.Module):
    def __init__(self, dim):
        super(MSCM,self).__init__()
        
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2))
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0))

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3))
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0))

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5))
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0))
        self.conv2 = nn.Conv2d(dim*4, dim, 1)

    def forward(self, x):        

        attn_0 = self.conv0_1(x)
        attn_0 = self.conv0_2(attn_0)#第一分支

        attn_1_1= self.conv1_1(x)
        attn_1_1 = self.conv1_2(attn_1_1)#第二分支

        attn_2_1 = self.conv2_1(x)
        attn_2_1 = self.conv2_2(attn_2_1)#第三分支

        attn_1_2 = self.conv1_1(attn_0 + attn_1_1)
        attn_1_2 = self.conv1_2(attn_1_2)#第二分支第二部分
        
        attn_2_2 = self.conv1_1(attn_1_2 + attn_2_1)
        attn_2_2 = self.conv1_2(attn_2_2)#第二分支第二部分


       
        attn = torch.cat((attn_0, attn_1_2, attn_2_2, x), 1)#concat

        attn = self.conv2(attn)#1*1卷积

        return attn
#全局特征融合
class GFFM1(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3,out_channels):#64 128 256 64
        super(GFFM1,self).__init__()
        #下采样
        self.Downconv = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv = nn.Conv2d(in_channels1, out_channels, 1)
        #上采样
        self.Upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels2, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels3, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.MSCM = MSCM(out_channels)

        self.conv1 = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, e2,e3,e4):        

        e2 = self.conv(e2)#64-64
        e3 = self.Upconv1(e3)#128-64
        e4 = self.Upconv2(e4)#256-64
        

        x = torch.cat((e2,e3,e4),1)
        x = self.conv1(x)

        return x
class GFFM2(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3,out_channels):#64 128 256 128
        super(GFFM2,self).__init__()
        #下采样
        self.Downconv = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv = nn.Conv2d(in_channels2, out_channels, 1)
        #上采样
        self.Upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels3, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Upconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels3, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.MSCM = MSCM(out_channels)

        self.conv1 = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, e2,e3,e4):        

        e2 = self.Downconv(e2)#64-64
        e3 = self.conv(e3)#128-64
        e4 = self.Upconv1(e4)#256-64
        

        x = torch.cat((e2,e3,e4),1)
        x = self.conv1(x)
        x = self.MSCM(x)

        return x
    
class GFFM3(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3,out_channels):#64 128 256 256
        super(GFFM3,self).__init__()
        #下采样
        self.Downconv1 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Downconv2 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        
        self.conv = nn.Conv2d(in_channels3, out_channels, 1)
        
        self.MSCM = MSCM(out_channels)

        self.conv1 = nn.Conv2d(out_channels * 3, out_channels, 1)

    def forward(self, e2,e3,e4):        

        e2 = self.Downconv2(e2)#64-256
        e3 = self.Downconv1(e3)#128-256
        e4 = self.conv(e4)#256-256
        

        x = torch.cat((e2,e3,e4),1)
        x = self.conv1(x)
        x = self.MSCM(x)

        return x



#下采样
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
           # nn.Conv2d(in_channels(输入数据的通道), out_channels（卷积产生的通道）,3（卷积核）,1（卷积步长）,1（填充）,padding_mode='reflect',bias=False),
           # padding_mode='reflect':填充映射，不填充0。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
#上采样
class DecoderBlock1(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock1,self).__init__()
        
        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.relu2 = nonlinearity        

    def forward(self, x):       
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x
    
   
#解码器
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.down1 = DoubleConv(filters[0], filters[1]) #64-128
        self.down2 = DoubleConv(filters[1], filters[2])
        self.down3 = DoubleConv(filters[1], filters[2]) #128-256

        self.up1 = DecoderBlock1(filters[2], filters[1]) #256-128
        self.up2 = DecoderBlock1(filters[1], filters[0]) #128-64
        self.up3 = DecoderBlock1(filters[1], filters[0]) #128-64

        self.gffm1 = GFFM1(filters[0],filters[1], filters[2], filters[0])#64
        self.gffm2 = GFFM2(filters[0],filters[1], filters[2], filters[1])#128
        self.gffm3 = GFFM3(filters[0],filters[1], filters[2], filters[2])#256

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        d5 = self.dblock(e4)

        #GFFM
        e5 = self.gffm1(e1,e2,e3)#64
        e6 = self.gffm2(e1,e2,e3)#128
        e7 = self.gffm3(e1,e2,e3)#256        

        # Decoder
        d4 = self.decoder4(d5) + e7
        d3 = self.decoder3(d4) + e6
        d2 = self.decoder2(d3) + e5
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)





