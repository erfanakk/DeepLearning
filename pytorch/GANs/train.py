import torch
import  data_setup

import torch.nn as nn
from  model_builder import Discriminator, Generator, init_weight 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys

#gper parametr
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 5 #number of epochs #TODO
z_dim = 100
image_channel = 1
LEARNING_RATE = 2e-4
batch_size = 128
n_feature = 64


dataloader = data_setup.creat_dataset(batch_size)

dis = Discriminator(in_channels=1, n_feature=n_feature).to(device)
generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)
init_weight(dis)
init_weight(generat)
fixed_size = torch.randn(size=(32, z_dim, 1, 1)).to(device)
fake_writer = SummaryWriter(f'runs/DCGAN/fake')
real_writer = SummaryWriter(f'runs/DCGAN/real')
step = 0



loss_fn = nn.BCELoss()
dic_opt = torch.optim.Adam(dis.parameters() , lr=LEARNING_RATE, betas=(0.5, 0.999))
gen_opt = torch.optim.Adam(generat.parameters() , lr=LEARNING_RATE, betas=(0.5, 0.999))
 

for epoch in  range(NUM_EPOCHS):
    for batch_idx , (real, _) in enumerate(dataloader):
        dis.train()
        generat.train()
        real = real.to(device)
        noise = torch.randn(size=(batch_size,z_dim,1,1))
        fake = generat(noise)
        
        #train dicsriminator MAX log(D(x)) + log(1-D(G(z)))
        disc_real = dis(real).reshape(-1)
        disc_fake = dis(fake).reshape(-1)

        lossDR= loss_fn(disc_real, torch.ones_like(disc_real))
        lossDF= loss_fn(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossDF + lossDR) / 2

        dis.zero_grad()
        lossD.backward(retain_graph=True) 
        dic_opt.step()
        


        #train generator min log(1-D(G(z)))

        output = dis(noise).reshape(-1)
        lossG = loss_fn(output,  torch.ones_like(output))
        generat.zero_grad()
        lossG.backward()
        gen_opt.step()




        if batch_idx == 0:
            dis.eval()
            generat.eval()
            print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
            with torch.no_grad():
                fake = generat(noise).reshape(-1, 1, 28, 28).to(device)
                data = real.reshape(-1, 1, 28, 28).to(device)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                fake_writer.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_writer.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1    




