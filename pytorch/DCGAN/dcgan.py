import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sys
import time





def save_checkpoint(stete , filname='DCGANs.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , dic: torch.nn.modules , gen: torch.nn.modules):
    print('-- loading model --')
    dic.load_state_dict(checkpoint['dic_dict'])
    gen.load_state_dict(checkpoint['gen_dict'])


epochs = 20
img_size = 64
bc_size = 32
z_dim = 100


device = "cuda" if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



data = datasets.MNIST(root='mnist/', transform=transform)
dataloader = DataLoader(data, batch_size=bc_size)


def init_w_model(model: torch.nn.Module()):
    for m in model.parameters():
        nn.init.normal_(m, 0, 0.02)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.feature = 64
        self.in_channel = in_channels  
        self.dic = nn.Sequential(
                *self._block(self.in_channel, self.feature, 4, stride=2, padding=1, bias=False),
                *self._block(self.feature, self.feature*2, 4, stride=2, padding=1, bias=False),
                *self._block(self.feature*2, self.feature*4, 4, stride=2, padding=1, bias=False),
                *self._block(self.feature*4, self.feature*8, 4, stride=2, padding=1, bias=False),
                nn.Conv2d(self.feature*8, out_channels=1, kernel_size=4,bias=False),
                nn.Sigmoid()
        )
        


    def _block(self, in_channels,out_channels,kernel_size,stride,padding,bias):
        block = nn.Sequential(
            nn.Conv2d(in_channels= in_channels,out_channels= out_channels,kernel_size= kernel_size,stride= stride,padding= padding,bias= bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block
    
    
    def forward(self,x):
        return self.dic(x)
        


class Generate(nn.Module):
    def __init__(self,out_channels=1):
        super().__init__()
        self.feature = 64
        self.gen = nn.Sequential(
            *self._block(in_channels=100, out_channels=self.feature*8, kernel_size=4, stride=1, padding=0),
            *self._block(in_channels=self.feature*8, out_channels=self.feature*4, kernel_size=4, stride=2,padding=1),
            *self._block(in_channels=self.feature*4, out_channels=self.feature*2, kernel_size=4, stride=2,padding=1),
            *self._block(in_channels=self.feature*2, out_channels=self.feature, kernel_size=4, stride=2,padding=1),
            nn.ConvTranspose2d(in_channels=self.feature, out_channels=1, kernel_size=4,stride=2,padding=1,bias=0),
            nn.Tanh()      
        )



    def _block(self,in_channels,out_channels,kernel_size,stride,padding,**kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,stride,padding,bias=False,**kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.gen(x)



gen = Generate().to(device)
dic = Discriminator().to(device)

init_w_model(gen)
init_w_model(dic)
fixed_noise = torch.randn(size=(bc_size,z_dim,1,1)).to(device)

loss_fn = nn.BCELoss()
D_opt = torch.optim.Adam(dic.parameters(), lr=0.001, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))

fake_writer = SummaryWriter(f'runs/dcgan/fake')
real_writer = SummaryWriter(f'runs/dcgan/real')
step = 0





for epoch in range(epochs):
    
    if (epoch+1) % 5 == 0:
        checkpoint = {
        'gen_dict' : gen.state_dict(),
        'dic_dict' : dic.state_dict()
        }
        save_checkpoint(stete=checkpoint)

    
    tic = time.time()
    
    for batch_idx, (real,_) in enumerate(dataloader):
        

    

        gen.train()
        dic.train()    

        noise = torch.randn(size=(bc_size,z_dim,1,1)).to(device)
        real = real.to(device)
        fake = gen(noise)

        dic_real = dic(real).reshape(-1)
        dic_fake = dic(fake).reshape(-1)

        lossDR = loss_fn(dic_fake, torch.zeros_like(dic_fake).to(device))
        lossDF = loss_fn(dic_real, torch.ones_like(dic_real).to(device))
        lossD = (lossDF + lossDR) / 2

        D_opt.zero_grad()
        lossD.backward(retain_graph=True)
        D_opt.step()


        out = dic(fake).reshape(-1)
        lossG = loss_fn(out, torch.ones_like(out).to(device))

        G_opt.zero_grad()
        lossG.backward()
        G_opt.step()

        if batch_idx == 1870 :
            print(
                        f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            toc = time.time()
            t = toc - tic
            print(f'for {epoch} wait {t}s')

            with torch.inference_mode():
                gen.eval()
                dic.eval()
                z = torch.randn(size=(bc_size,z_dim,1,1)).to(device)
                fake = gen(z).reshape(-1,1,64,64).to(device)
                real = real.reshape(-1,1,64,64).to(device)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                real_writer.add_image('mnist dcgan real images', img_grid_real, global_step=step)
                fake_writer.add_image('mnist dcgan fake images', img_grid_fake, global_step=step)

                step += 1


    

