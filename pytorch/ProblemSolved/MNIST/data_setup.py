import torchvision
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from utils import get_mean_std

if_lenet = False

def creat_dataset():
    if if_lenet:
        transform_train = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
         ])
        transform_test = transforms.Compose([
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            
         ])
    else:
            transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))   
         ])
            transform_test = transforms.Compose([
            transforms.ToTensor()   
          ])


    train_dataset = datasets.MNIST(root="MNIST/", train=True, transform=transform_train, download=True)
    test_dataset = datasets.MNIST(root="MNIST/", train=False, transform=transform_test, download=True)

    class_names = train_dataset.classes
    
    trainset = DataLoader(train_dataset , batch_size= 32 , shuffle= True)
    testset = DataLoader(test_dataset , batch_size= 32 , shuffle= True)
    
    return trainset , testset , class_names





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
