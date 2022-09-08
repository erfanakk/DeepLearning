import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


def creat_dataset(num_batch):
    transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]) 
    
    dataset = datasets.MNIST(root="MNIST/",  transform=transform_train, download=True)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=num_batch)
    return dataloader





def test():
    train , test , class_names= creat_dataset()
    imgesT , label = next(iter(train))
    imges , label = next(iter(test))
    
    print(class_names)
    
    print(f'we have {len(train)} batch and {imgesT.shape[0] * (len(train))} images in each batch in train dataset')
    print(f'we have {len(test)} batch and {imges.shape[0] * (len(test))} images in each batch in test dataset')
    print(f'shape of image c= {imgesT.shape[1]} h= {imgesT.shape[2]} w= {imgesT.shape[3]}')
    # mean, std = get_mean_std(train)
    # print(f'this is mean {mean} and this is std {std}')
    #mean -----> 0.1307
    # std ---->  0.3081

if '__main__' == __name__:
    test()