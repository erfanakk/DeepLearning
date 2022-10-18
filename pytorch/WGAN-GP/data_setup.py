import torch 
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms , datasets







def datasetmnist(bc_size, img_size , mnist , cleb):

    if cleb:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])



        data = datasets.CelebA(root='cleba/', transform=transform)
        dataloader = DataLoader(data, batch_size=bc_size, shuffle=True)
    
    if mnist:
        transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
     ])



        data = datasets.MNIST(root='mnist/', transform=transform, download=True)
        dataloader = DataLoader(data, batch_size=bc_size, shuffle=True)
          
    return dataloader




    
if __name__ == "__main__":
    data = datasetmnist(32,64, mnist=False, cleb=True)
    (img , labels) = next(iter(data))
    print(img.shape)

