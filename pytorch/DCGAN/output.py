import utils
import torch
import torch.nn as nn
from model_builder import Generator, Discriminator
import torchvision
import sys






z_dim = 100
image_channel = 1
n_feature = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)
dis = Discriminator(in_channels=1, n_feature=n_feature).to(device)

utils.load_checkpoint(torch.load('DCMODEL.pth.tar' , map_location=torch.device('cpu')), modelGEN=generat, modelDIS=dis)

for i in range(3):
    noise = torch.randn(size=(32, z_dim, 1, 1)).to(device)
    muls = [-1,-5,0.5,-0.15]
    for mul in muls:
        z = noise*mul
        images = generat(z)
        #print(z[:,1,:,:])

        img_grid = torchvision.utils.make_grid(images[:32], normalize=True)
        torchvision.utils.save_image(img_grid , f'generat_out{i}{mul}.png')
    print('done')
