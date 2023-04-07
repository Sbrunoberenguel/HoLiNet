import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from collections import OrderedDict

from ResNet50_Equi import EquiConv2d,Encoder

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}

#Helping function in case of bad initialization of the network
def xavier_init(m):
    classname = m.__class__.__name__
    if (classname.find('EquiConv2d') != -1):
        nn.init.xavier_normal(m.weight.data)

def load_resnet50():
    model = Encoder()
    model_dict = model.state_dict()
    # Load pre-trained weights
    url_state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                            progress=True)
    # Filter out unnecessary keys
    state_dict = {'encoder.'+k: v for k, v in url_state_dict.items() if ('encoder.'+k) in model_dict}
    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # Load the new state dict
    model.load_state_dict(model_dict)
    return model

# Decoder of Holinet
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ##Decoder
        self.d_2x_ec = nn.Sequential(EquiConv2d(2048,512,kernel=3,stride=1,bias=True),nn.ReLU())
        self.d_2x = nn.UpsamplingBilinear2d(scale_factor=2)
        #
        self.d_4x_ec = nn.Sequential(EquiConv2d(512+1024,256,kernel=3,stride=1,bias=True),nn.ReLU())
        self.d_4x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.output4X_likelihood = EquiConv2d(256,2,kernel=3,stride=1,bias=True)
        #
        self.d_8x_ec = nn.Sequential(EquiConv2d(256+512+2,128,kernel=3,stride=1,bias=True),nn.ReLU())
        self.d_8x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.output8X_likelihood = EquiConv2d(128,2,kernel=3,stride=1,bias=True)
        #
        self.d_16x_ec = nn.Sequential(EquiConv2d(128+256+2,64,kernel=3,stride=1,bias=True),nn.ReLU())
        self.d_16x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.output16X_likelihood = EquiConv2d(64,2,kernel=3,stride=1,bias=True)
        #
        self.d_16x_conv1 = nn.Sequential(EquiConv2d(64+64+2,64,kernel=5,stride=1,bias=True),nn.ReLU())
        self.d_32x = nn.UpsamplingBilinear2d(scale_factor=1)
        self.output_likelihood = EquiConv2d(64,2,kernel=3,stride=1,bias=True)

        ## In case of training from scratch and stability and convergence problems
        ##   uncomment the next line for weights initialization (may do not solve the problem)
        
        # self.apply(xavier_init)

    def forward(self,x):
        bn_conv1,res2c_relu,res3d_relu,res4f_relu,bn5c_branch2c = x
        d_2x_ec = self.d_2x_ec(bn5c_branch2c)
        d_2x = self.d_2x(d_2x_ec)
        #
        d_4x_ec = self.d_4x_ec(torch.cat([d_2x,res4f_relu],dim=1))
        d_4x = self.d_4x(d_4x_ec)
        output4X_likelihood = self.output4X_likelihood(d_4x)
        #
        d_8x_ec = self.d_8x_ec(torch.cat([d_4x,res3d_relu,output4X_likelihood],dim=1))
        d_8x = self.d_8x(d_8x_ec)
        output8X_likelihood = self.output8X_likelihood(d_8x)
        #
        d_16x_ec = self.d_16x_ec(torch.cat([d_8x,res2c_relu,output8X_likelihood],dim=1))
        d_16x = self.d_16x(d_16x_ec)
        output16X_likelihood = self.output16X_likelihood(d_16x)
        #
        d_16x_conv1 = self.d_16x_conv1(torch.cat([d_16x,bn_conv1,output16X_likelihood],dim=1))
        d_32x = self.d_32x(d_16x_conv1)
        output_likelihood = self.output_likelihood(d_32x)

        return [output_likelihood,output16X_likelihood,output8X_likelihood,output4X_likelihood]


class HoLiNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485,0.456,0.406])[None, :, None, None])
    x_std  = torch.FloatTensor(np.array([0.229,0.224,0.225])[None, :, None, None])

    def __init__(self,pretrained_backbone=False):
        super().__init__()
        if pretrained_backbone:
            # Load ResNet50 with pre-trained weights on ImageNet
            self.backbone = load_resnet50()
        else:
            # Loads ResNet50 with RANDOM weights (not trained)
            self.backbone = Encoder()

        self.layout_detector = Decoder()

    def forward(self,x,train=False):
        x = self.prepare_x_(x)
        feature_list = self.backbone(x)
        out = self.layout_detector(feature_list)
        if train:
            # Returns the output and early predictions (need a sigmoid for evaluating)
            return out[0],out[1:]
        else:
            # Returns output through a sigmoid funcion
            return F.sigmoid(out[0])

    def prepare_x_(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

def save_weights(net,args,path):
    state_dict = OrderedDict({
                'args': args.__dict__,
                'state_dict': net.state_dict()})
    torch.save(state_dict,path)
    
def load_weigths(args):
    stt_dict = torch.load(args.pth,map_location='cpu')
    state_dict = stt_dict['state_dict']
    net = HoLiNet()
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict)
    return net

if __name__=='__main__':
    net = HoLiNet(True)
    print('Network created')
    device = torch.device('cuda')
    batch = 2
    ch,h,w = 3,256,512 #,1024
    dummy = torch.rand((batch,ch,h,w))
    out = net(dummy)
    print(out.shape)