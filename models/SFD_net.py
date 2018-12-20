import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import os

class L2Norm(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None,:,None,None]
        return scale * x

def STlayer(x,r):
    r=int(r)
    pixel_shuffle=nn.PixelShuffle(r)
    channels=x.shape[1]
    O=torch.Tensor(x.shape[0], channels//(r**2), x.shape[2]*r, x.shape[3]*r)
    O=O.cuda()
    #print(O.shape,x.shape)
    O=Variable(O, requires_grad=False)
    for i in range(0,channels,r**2):
        O[:,i%(r**2):i%(r**2)+1,:,:] = pixel_shuffle(x[:,i:i+r**2,:,:])
    #O = nn.ReLU(O,inplace=True)
    return O



class SFDNet(nn.Module):
    def __init__(self, phase, base, extras, head):
        super(SFDNet, self).__init__()
        self.phase = phase
        # vgg network
        self.base = nn.ModuleList(base)
        self.conv3_Norm=L2Norm(256,10)
        self.conv4_Norm=L2Norm(512,8)
        self.conv5_Norm=L2Norm(512,5)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(16):
            x = self.base[k](x)
        s=self.conv3_Norm(x)
        sources.append(s)
        
        for k in range(16,23):
            x = self.base[k](x)
        s=self.conv4_Norm(x)
        sources.append(s)
        
        for k in range(23,30):
            x= self.base[k](x)
        s=self.conv5_Norm(x)
        sources.append(s)
        
        for k in range(30,len(self.base)):
            x= self.base[k](x)
        sources.append(x)
        # apply extra layers and cache source layer outputs
        x = self.extras[0](x)
        x = self.extras[1](x)
        x = self.extras[2](x)
        x = self.extras[3](x)
        sources.append(x)
        x = self.extras[4](x)
        x = self.extras[5](x)
        x = self.extras[6](x)
        x = self.extras[7](x)
        sources.append(x)


        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, 2)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '640': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras():
    # Extra layers added to VGG for feature scaling
    layers = []
    conv6_1 = nn.Conv2d(1024, 256, kernel_size=1)
    conv6_2 = nn.Conv2d(256,512, kernel_size=3,stride=2,padding=1)
    conv7_1 = nn.Conv2d(512, 128, kernel_size=1)
    conv7_2 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1)
    layers += [conv6_1, nn.ReLU(inplace=True), conv6_2, nn.ReLU(inplace=True),
               conv7_1, nn.ReLU(inplace=True), conv7_2, nn.ReLU(inplace=True)]
    return layers

def multibox(vgg):
    loc_layers = []
    conf_layers = []

    loc_layers = loc_layers+[nn.Conv2d(vgg[14].out_channels, 4, kernel_size=3, padding=1)]
    conf_layers = conf_layers+[nn.Conv2d(vgg[14].out_channels, 2, kernel_size=3, padding=1)]
    loc_layers = loc_layers+[nn.Conv2d(vgg[21].out_channels, 4, kernel_size=3, padding=1)]
    conf_layers = conf_layers+[nn.Conv2d(vgg[21].out_channels, 2, kernel_size=3, padding=1)]
    loc_layers = loc_layers+[nn.Conv2d(vgg[28].out_channels, 4, kernel_size=3, padding=1)]
    conf_layers = conf_layers+[nn.Conv2d(vgg[28].out_channels, 2, kernel_size=3, padding=1)]
    loc_layers = loc_layers+[nn.Conv2d(vgg[33].out_channels, 4, kernel_size=3, padding=1)]
    conf_layers = conf_layers+[nn.Conv2d(vgg[33].out_channels, 2, kernel_size=3, padding=1)]

    #loc_layers += [nn.Conv2d(128, 4, kernel_size=3, padding=1)]
    #conf_layers += [nn.Conv2d(128, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(512, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, 2, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2, kernel_size=3, padding=1)]

    return (loc_layers, conf_layers)



def build_net(phase):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    return SFDNet(phase, vgg(base[str(640)],3), add_extras(), multibox(vgg(base[str(640)],3)))
