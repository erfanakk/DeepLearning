
import torch
import torch.nn as nn



def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

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


def save_checkpoint(stete , filname='C-GAN.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , model: torch.nn.modules , optimizer: torch.optim.Optimizer):
    print('-- loading model --')
    model.load_state_dict(checkpoint['state_dic'])
    optimizer.load_state_dict(checkpoint['optimizer'])

