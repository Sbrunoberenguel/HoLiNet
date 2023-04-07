import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import deform_conv2d, DeformConv2d
from torch.hub import load_state_dict_from_url


'''
In this file we have re-coded ResNet-50 to use with Equirectangular convolutions
It is a 'hard-coded' architecture to replace ResNet model from pytorch
You can also load pre-trained weights from Pytorch as in the standard model, 
  but it will only use Equirectangular convolutions
'''

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}

#Dictionary of pre-computed kernel distortions
Dict_offsets = torch.load('offset_HoliNet.pt')

# Implementation of deformable convolutions in a TensorFlow like way
class EquiConv2d(DeformConv2d):
    '''
    Original code from Daniel Kurnia Suhendro (https://github.com/palver7)
    Adapted to our implementation for pre-computed kernel distortion
    '''

    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel, stride, 0, dilation, groups, bias)
        self.stride = stride
        self.kernel = kernel

    def forward(self, x):
        device = x.device
        bs = x.size()[0]
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride, self.stride
        key = ((ih//sh,iw//sw),(self.kernel,self.kernel),(self.stride,self.stride))
        offset = Dict_offsets[key].to(device)
        offset = torch.cat([offset for _ in range(bs)],dim=0)
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        if x.shape[0] != offset.shape[0] :
            sizediff = offset.shape[0] - x.shape[0]
            offset = torch.split(offset,[x.shape[0],sizediff],dim=0)
            offset = offset[0]
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation) 


class Bottleneck(nn.Module):
    '''
    Bottleneck structure for each Resnet's layer
    When 1x1 convolution used (no distortion in the kernel), we use standard Conv2d for faster inference time 
    '''
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, DropOut=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes // 4
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.DropOut = DropOut
        self.conv1 = nn.Conv2d(inplanes , width,  kernel_size = 1 , stride = 1)
        self.bn1 = norm_layer(width)
        self.conv2 = EquiConv2d(width   , width,  kernel = 3 , stride = stride)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width    , planes, kernel_size = 1 , stride = 1)
        self.bn3 = norm_layer(planes)
        self.dropout = nn.Dropout(0.3, inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.DropOut == True:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.DropOut == True:
            out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    '''
    This is our own implementation of ResNet-50's architecture to apply Equirectangular convolutions.
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        self.initPlanes = 64
        self.layer1Planes = 256
        self.layer2Planes = 512
        self.layer3Planes = 1024
        self.layer4Planes = 2048
        self.conv1 = EquiConv2d(3, self.initPlanes, kernel=7, stride = 2)
        self.bn1   = nn.BatchNorm2d(self.initPlanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,dilation=1,ceil_mode=False)

        self.layer1 = nn.Sequential(
            Bottleneck(self.initPlanes, self.layer1Planes, stride=1, downsample=nn.Sequential(
                nn.Conv2d(self.initPlanes,self.layer1Planes, kernel_size = 1, stride = 1),
                nn.BatchNorm2d(self.layer1Planes)
            )),
            Bottleneck(self.layer1Planes, self.layer1Planes, stride=1, downsample=None),
            Bottleneck(self.layer1Planes, self.layer1Planes, stride=1, downsample=None)
        )

        self.layer2 = nn.Sequential(
            Bottleneck(self.layer1Planes, self.layer2Planes, stride=2, downsample=nn.Sequential(
                nn.Conv2d(self.layer1Planes,self.layer2Planes, kernel_size = 1, stride = 2),
                nn.BatchNorm2d(self.layer2Planes)
            )),
            Bottleneck(self.layer2Planes, self.layer2Planes, stride=1, downsample=None),
            Bottleneck(self.layer2Planes, self.layer2Planes, stride=1, downsample=None),
            Bottleneck(self.layer2Planes, self.layer2Planes, stride=1, downsample=None)
        )

        self.layer3 = nn.Sequential(
            Bottleneck(self.layer2Planes, self.layer3Planes, stride=2, downsample=nn.Sequential(
                nn.Conv2d(self.layer2Planes,self.layer3Planes, kernel_size = 1, stride = 2),
                nn.BatchNorm2d(self.layer3Planes)
            )),
            Bottleneck(self.layer3Planes, self.layer3Planes, stride=1, downsample=None),
            Bottleneck(self.layer3Planes, self.layer3Planes, stride=1, downsample=None),
            Bottleneck(self.layer3Planes, self.layer3Planes, stride=1, downsample=None),
            Bottleneck(self.layer3Planes, self.layer3Planes, stride=1, downsample=None),
            Bottleneck(self.layer3Planes, self.layer3Planes, stride=1, downsample=None)
        )

        self.layer4 = nn.Sequential(
            Bottleneck(self.layer3Planes, self.layer4Planes, stride=2, downsample=nn.Sequential(
                nn.Conv2d(self.layer3Planes,self.layer4Planes, kernel_size=1, stride=2),
                nn.BatchNorm2d(self.layer4Planes)
            )),
            Bottleneck(self.layer4Planes, self.layer4Planes, stride=1, downsample=None),
            Bottleneck(self.layer4Planes, self.layer4Planes, stride=1, downsample=None, DropOut=True)
        )

    def forward(self,x):
        feature_list = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        feature_list.append(out)
        out = self.maxpool(out)

        out = self.layer1(out); feature_list.append(out)
        out = self.layer2(out); feature_list.append(out)
        out = self.layer3(out); feature_list.append(out)
        out = self.layer4(out); feature_list.append(out)
        return feature_list

class ResNet50_Equi(nn.Module):

    def __init__(self):
        super(ResNet50_Equi, self).__init__()
        self.encoder = Encoder()
    
    def forward(self,x):
        out = self.encoder(x)
        return out


def load_resnet50():
    model = ResNet50_Equi()
    model_dict = model.state_dict()
    # 0. Load pre-trained weights
    url_state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                            progress=True)
    # 1. filter out unnecessary keys
    # state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    state_dict = {'encoder.'+k: v for k, v in url_state_dict.items() if ('encoder.'+k) in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    resnet = load_resnet50().to('cuda')
    dummy = torch.randn(4,3,512,1024)
    FL = resnet(dummy.to('cuda'))
    out = FL[-1]
    print(out.shape)
    print('Done')