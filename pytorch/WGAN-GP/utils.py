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
 


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), requires_grad=True).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty