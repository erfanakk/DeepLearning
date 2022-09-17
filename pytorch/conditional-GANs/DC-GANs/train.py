import torch
import  data_setup

import torch.nn as nn
from  model_builder import Discriminator, Discriminator, init_weight 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys
from utils import gradient_penalty, save_checkpoint
#gper parametr

import argparse
import sys


import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--epoch', action='store', type=int, required=True)
my_parser.add_argument('--save', action='store', type=bool)
args = my_parser.parse_args()



torch.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = args.epoch #number of epochs #TODO
z_dim = 100
image_channel = 1
LEARNING_RATE = 1e-4
batch_size = 64
n_feature = 16
critic_iter = 5
img_size = 64
num_class =10
gen_embedding = 100
lambda_gp = 10
save_model = args.save



dataloader = data_setup.creat_dataset(batch_size)

dis = Discriminator(in_channels=1, n_feature=n_feature, img_size=img_size, num_classes=num_class).to(device)
#z_dim, n_feature, img_chan, num_classes, img_size, embed_size)
generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel, img_size=img_size, num_classes=num_class, embed_size= gen_embedding, ).to(device)
init_weight(dis)
init_weight(generat)

fixed_size = torch.randn(size=(32, z_dim, 1, 1)).to(device)

fake_writer = SummaryWriter(f'runs/DCGAN/fake')
real_writer = SummaryWriter(f'runs/DCGAN/real')
step = 0




dic_opt = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
gen_opt = torch.optim.Adam(generat.parameters() , lr=LEARNING_RATE, betas=(0.0, 0.9))



for epoch in  range(NUM_EPOCHS):

    for batch_idx , (real, labels) in enumerate(dataloader):
        dis.train()
        generat.train()
        real = real.to(device)
        labels = labels.to(device)
        for _ in range(critic_iter):

        
            noise = torch.randn(size=(batch_size,z_dim,1,1)).to(device)
            fake = generat(noise, labels)
        
        #train dicsriminator
            
            disc_real = dis(real, labels).reshape(-1)
            disc_fake = dis(fake, labels).reshape(-1)
            gp = gradient_penalty(dis,labels ,real, fake, device=device)
            lossD = (
                -(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp
            )
            

            dic_opt.zero_grad()
            lossD.backward(retain_graph=True) 
            dic_opt.step()



        #train generator min log(1-D(G(z)))

        output = dis(fake, labels).reshape(-1)
        lossG = -torch.mean(output)
        gen_opt.zero_grad()
        lossG.backward()
        gen_opt.step()

        
        if save_model:
            if epoch % 5 == 0:
                checkpoint = {
                            'state_dic' : generat.state_dict(),
                            'optimizer' : gen_opt.state_dict()
                            }
                utils.save_checkpoint(stete=checkpoint)




        if batch_idx == 0:

            print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
            with torch.no_grad():
                dis.eval()
                generat.eval()
               
                fake = generat(noise, labels)
               
                data = real
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)

                fake_writer.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                real_writer.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1    




