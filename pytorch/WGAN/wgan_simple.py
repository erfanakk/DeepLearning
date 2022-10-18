import torch
import torch.nn as nn
import numpy as np








class Critic(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=int(np.prod(img_shape)), out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x 









class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=28):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x










if __name__ == '__main__':
    img_shape = (1,28,28)
    
    z_dim = 100
    gen = generator()
    z = torch.randn(size=(5,100))
    
    out = gen(z)
    print(out.shape)