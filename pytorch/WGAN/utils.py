import torch
import time
import torch.nn as nn









def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)





def to_img(x):
    x = x.clamp(0,1)
    return x

def save_checkpoint(stete , filname='wgan_simple.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , dic: torch.nn.modules , gen: torch.nn.modules):
    print('-- loading model --')
    dic.load_state_dict(checkpoint['dic_dict'])
    gen.load_state_dict(checkpoint['gen_dict'])
 