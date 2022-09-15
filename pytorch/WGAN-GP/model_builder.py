
import torch
import torch.nn as nn







torch.manual_seed(42)



class Critic(nn.Module):
    
    def __init__(self , in_channels, n_feature):
        super().__init__()
        #input 3*64*64
        self.dic = nn.Sequential(
            nn.Conv2d(in_channels, n_feature, kernel_size=4, stride=2, padding=1), #32*32
            nn.LeakyReLU(0.2),
            self._block(n_feature, n_feature*2, kernel_size=4, stride=2, padding=1),#16*16
            self._block(n_feature*2, n_feature*4, kernel_size=4, stride=2, padding=1),#8*8
            self._block(n_feature*4, n_feature*8, kernel_size=4, stride=2, padding=1),#4*4
            nn.Conv2d(n_feature*8, 1,kernel_size=4, stride=2, padding=0), #1,1
         )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.dic(x)    

class Generator(nn.Module):
    def __init__(self, z_dim, n_feature, img_chan):
        super().__init__()
        
        #input ----> N*100*1*1
        self.gen = nn.Sequential(
            self._block(z_dim, n_feature*16, kernel_size=4, stride=1, padding=0),# N,(n_feature*16)*4*4
            self._block(n_feature*16, n_feature*8, kernel_size=4, stride=2, padding=1),#8*8
            self._block(n_feature*8, n_feature*4, kernel_size=4, stride=2, padding=1),#16*16
            self._block(n_feature*4, n_feature*2, kernel_size=4, stride=2, padding=1),#32*32
            nn.ConvTranspose2d(n_feature*2, img_chan, kernel_size=4, stride=2, padding=1),#N*3*64*64
            nn.Tanh(),#[-1, 1]
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x)




def init_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.00, 0.02)



if __name__ == '__main__':
    vector = torch.rand(size=(1,100,1,1))
    gen = Generator(z_dim=100, n_feature=8, img_chan=3)
    init_weight(gen)
    print(gen(vector).shape) 
    img = torch.rand(size=(1,3,64,64))
    dis = Critic(in_channels=3, n_feature=8)
    init_weight(dis)
    print(dis(img).shape)


