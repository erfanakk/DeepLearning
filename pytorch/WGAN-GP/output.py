import utils
import torch
import torch.nn as nn
from model_builder import Generator
import torchvision






z_dim = 100
image_channel = 1
n_feature = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)

utils.load_checkpoint(torch.load('DCMODEL.pth.tar' , map_location=torch.device('cpu')), modelGEN=generat )


noise = torch.randn(size=(32, z_dim, 1, 1)).to(device)

images = generat(noise)


img_grid = torchvision.utils.make_grid(fake[:32], normalize=True)
torchvision.utils.save_image(img_grid , f'generat_out.png')
