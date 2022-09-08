
import torch
import torch.nn as nn







torch.manual_seed(42)



class Discriminator(nn.Module):
    
    def __init__(self , in_channels):
        super().__init__()
        self.dic = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dic(x)    

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)









if '__main__' == __name__:
    img = torch.rand(size=(100,))
    #print(img)
    model = Generator(z_dim=100, img_dim=28*28)
    print(model(img).shape)
