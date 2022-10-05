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
import torch.nn.functional as F



def save_checkpoint(stete , filname='CGANs.pth.tar'):
    print(f'save the model in {filname}')
    torch.save(stete, f=filname)

def load_checkpoint( checkpoint , dic: torch.nn.modules , gen: torch.nn.modules):
    print('-- loading model --')
    dic.load_state_dict(checkpoint['dic_dict'])
    gen.load_state_dict(checkpoint['gen_dict'])


epochs = 20
img_size = 28
bc_size = 32
z_dim = 100
adam_lr = 0.0002
adam_beta = 0.5

device = "cuda" if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])



data = datasets.MNIST(root='mnist/', transform=transform)
dataloader = DataLoader(data, batch_size=bc_size, shuffle=True)







class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super().__init__()

        #input zdim --> (N,100,1,1) ----> (N,128,3,3)    
        self.layer1 =nn.Sequential( 
                                    nn.ConvTranspose2d(in_channels=100, out_channels=128, kernel_size=3,
                                                    stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU()
        )
        
        #class --->(N,10,1,1) ------>(N,128,3,3)
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=10, out_channels=128, kernel_size=3,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #input (N,256,3,3)----------> (N,1,28,28)
        self.layer12 = nn.Sequential(
            *self._block(in_channels=256, out_channels=128, kernel_size=3,
                         padding=0),

            *self._block(in_channels=128, out_channels=64,kernel_size=4,padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()            
        )
    


    def _block(self,in_channels,out_channels,kernel_size, padding,stride=2, bias=False,**kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,kernel_size, stride,padding=padding ,bias=bias, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x,y):
        #x = (N,100)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        # x = (N,100,1,1)
        x = self.layer1(x)
        #x = (N,128,3,3)

        #y = (N,10)
        y = y.view(y.shape[0],y.shape[1], 1,1)
        # y = (N,100,1,1)
        y = self.layer2(y)
        # y = (N,128,3,3)


        #concat x , y
        xy = torch.cat([x,y],dim=1)
        #xy = (N,256,3,3)

        xy = self.layer12(xy)

        #xy = (N,1,28,28)

        return xy

        






class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()


        #input image (N,1,28,28) ----> (N,32,14,14)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True)

        )

        # input label (N, 10:num_class, 28, 28) -----> (N,32,14,14)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=4,stride=2,padding=1,bias=False),
            nn.LeakyReLU(0.2,inplace=True)            
        )

        #concat layer1 , layer2 (N,64,14,14) ------> predict zreald or fake

        self.layer12 = nn.Sequential(

            *self._block(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            #out ---> (N,128,7,7)
            *self._block(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            #out ---> (N,256,3,3)
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3,stride=1,padding=0,bias=False),
            #out ---> (N,1,1,1)
            nn.Sigmoid()        
        )
    
    def _block(self, in_channels, out_channels,kernel_size,stride ,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2,inplace=True)
        )
    
    
    def forward(self,x,y):
        #x = images
        # x ---> (N,1,28,28)
        x = self.layer1(x)
        #x ---> (N,32,14,14)

        #y = labael
        #y ---> (N,10,28,28)
        y =  self.layer2(y)
        #y ----> (N,32,14,14)

        xy = torch.cat([x,y],dim=1)
        xy = self.layer12(xy)
        xy = xy.view(xy.shape[0], -1)
        return xy





# custom weights initialization
def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)






gen = Generator().to(device)
dic = Discriminator().to(device)

gen.apply(weights_init)
dic.apply(weights_init)



# We calculate Binary cross entropy loss
fn_loss = nn.BCELoss()
# Adam optimizer for generator 
optimizerG = torch.optim.Adam(gen.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))
# Adam optimizer for discriminator 
optimizerD = torch.optim.Adam(dic.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))

optimizerD = torch.optim.Adam(dic.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))

fake_writer = SummaryWriter(f'runs/Gan_c/fake')
step = 21



label_real = torch.ones((bc_size,1)).to(device)
label_fake = torch.zeros((bc_size,1)).to(device)
z_test = torch.randn(size=(100, z_dim)).to(device)





onehot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)

fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1
test_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]*10).type(torch.LongTensor)

test_Gy = onehot[test_y].to(device)












#train 
if __name__ == '__main__':
    load_checkpoint(torch.load('CGANs.pth.tar') , dic=dic, gen=gen)
    for epoch in range(epochs):
            
        if (epoch+1) % 5 == 0:
            checkpoint = {
            'gen_dict' : gen.state_dict(),
            'dic_dict' : dic.state_dict()
            }
            save_checkpoint(stete=checkpoint)



        for batch_idx , (images, labels) in enumerate(dataloader):
            tic = time.time()
            gen.train()
            dic.train()
            
            images = images.to(device)
            #label 
            img_y = fill[labels].to(device)
            
            D_real = dic(images,img_y)
            D_R_loss = fn_loss(D_real, label_real )
                    # create latent vector z from normal distribution 
            z = torch.randn(bc_size, z_dim).to(device)
            # create random y labels for generator
            y_gen = (torch.rand(bc_size, 1)*10).type(torch.LongTensor).squeeze()
            # convert genarator labels to onehot
            G_y = onehot[y_gen].to(device)
            # preprocess labels for feeding as y input in D
            # DG_y shape will be (batch_size, 10, 28, 28)
            DG_y = fill[y_gen].to(device)

            fake_img = gen(z,G_y)

            D_fake = dic(fake_img.detach(), DG_y)
            D_F_loss = fn_loss(D_fake, label_fake)

            D_loss = (D_F_loss + D_R_loss) / 2


            dic.zero_grad()
            D_loss.backward()
            optimizerD.step()


            out = dic(fake_img, DG_y)
            G_loss = fn_loss(out, label_real)

            gen.zero_grad()
            G_loss.backward()
            optimizerG.step()


            
            if batch_idx == (len(dataloader)-2) :
                print(
                            f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                            Loss D: {D_loss:.2f}, loss G: {G_loss:.4f}"
                )
                toc = time.time()
                t = toc - tic
                print(f'for {epoch} wait {t:.3f}s')

                with torch.inference_mode():
                    gen.eval()
                    dic.eval()
                    fake_test = gen(z_test, test_Gy).cpu()
                    # save images in grid of 10 * 10
                    torchvision.utils.save_image(fake_test, f"mnist_epoch_{epoch+1}.jpg", nrow=10, padding=0, normalize=True)
                    img_grid = torchvision.utils.make_grid(fake_test, normalize=True)
                    fake_writer.add_image('mnist cgan fake images', img_grid, global_step=step)

                    step += 1


            









