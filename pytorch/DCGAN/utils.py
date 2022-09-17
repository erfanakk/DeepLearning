
import torch
import torch.nn as nn



def save_checkpoint(stete , filname='DCMODEL.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , modelGEN: torch.nn.modules, modelDIS: torch.nn.modules , optimizer: torch.optim.Optimizer):
    print('-- loading model --')
    modelGEN.load_state_dict(checkpoint['state_dic_GEN'])
    modelDIS.load_state_dict(checkpoint['state_dic_DIS'])
    optimizer.load_state_dict(checkpoint['optimizer'])

