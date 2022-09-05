import torch
from pathlib import Path


def save_checkpoint(stete , filname='mymodel.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , model: torch.nn.modules , optimizer: torch.optim.Optimizer):
    print('-- loading model --')
    model.load_state_dict(checkpoint['state_dic'])
    optimizer.load_state_dict(checkpoint['optimizer'])




def get_mean_std(dataLoader):
    chanl_sum, chanl_squared_sum , num_batch = 0, 0, 0 
    for images, _ in dataLoader:
        chanl_sum += torch.mean(images , dim=[0,2,3])
        chanl_squared_sum += torch.mean(images**2 , dim=[0,2,3])
        num_batch += 1

    mean = chanl_sum/num_batch
    std = (chanl_squared_sum/num_batch - mean**2)**0.5

    return mean, std