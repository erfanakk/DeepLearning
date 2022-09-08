import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


def creat_dataset(num_batch):
    transform = transforms.Compose([ 
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]) 
    
    dataset = datasets.MNIST(root="MNIST/",  transform=transform, download=True)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=num_batch)
    return dataloader




