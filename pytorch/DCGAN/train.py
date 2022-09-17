import torch
import  data_setup
import utils
import torch.nn as nn
from  model_builder import Discriminator, Generator, init_weight 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys


import sys
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--epoch', action='store', type=int, required=True)
my_parser.add_argument('--save', action='store', type=bool)
my_parser.add_argument('--retrain', action='store', type=bool)

args = my_parser.parse_args()



torch.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = args.epoch #number of epochs #TODO
z_dim = 100
image_channel = 1
LEARNING_RATE = 2e-4
batch_size = 64
n_feature = 64
save_model = args.save
retrain = args.retrain

dataloader = data_setup.creat_dataset(batch_size)

dis = Discriminator(in_channels=1, n_feature=n_feature).to(device)
generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)
init_weight(dis)
init_weight(generat)

fixed_size = torch.randn(size=(32, z_dim, 1, 1)).to(device)

fake_writer = SummaryWriter(f'runs/DCGAN/fake')
real_writer = SummaryWriter(f'runs/DCGAN/real')
step = 0




dic_opt = torch.optim.Adam(dis.parameters() , lr=LEARNING_RATE, betas=(0.5, 0.999))
gen_opt = torch.optim.Adam(generat.parameters() , lr=LEARNING_RATE, betas=(0.5, 0.999))
loss_fn = nn.BCELoss()

if retrain:
    utils.load_checkpoint(torch.load('DCMODEL.pth.tar' , map_location=torch.device('cpu')), modelDIS=dis,modelGEN=generat , optimizer=gen_opt)



for epoch in  range(NUM_EPOCHS):
    for batch_idx , (real, _) in enumerate(dataloader):
        dis.train()
        generat.train()
        real = real.to(device)
        noise = torch.randn(size=(batch_size,z_dim,1,1)).to(device)
        fake = generat(noise)
        
        #train dicsriminator MAX log(D(x)) + log(1-D(G(z)))
        disc_real = dis(real).reshape(-1)
        disc_fake = dis(fake).reshape(-1)

        lossDR= loss_fn(disc_real, torch.ones_like(disc_real))
        lossDF= loss_fn(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossDF + lossDR) / 2

        dic_opt.zero_grad()
        lossD.backward(retain_graph=True) 
        dic_opt.step()
        


        #train generator min log(1-D(G(z)))

        output = dis(fake).reshape(-1)
        lossG = loss_fn(output,  torch.ones_like(output))
        gen_opt.zero_grad()
        lossG.backward()
        gen_opt.step()


        

        if (batch_idx % 50) == 0:

            print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
            with torch.no_grad():
                dis.eval()
                generat.eval()
               
                fake = generat(fixed_size)
               
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

    if save_model:
        if epoch % 5 == 0:
            checkpoint = {
                        'state_dic_GEN' : generat.state_dict(),
                        'state_dic_DIS' : dis.state_dict(),
                        'optimizer' : gen_opt.state_dict()
                        }
            utils.save_checkpoint(stete=checkpoint)
        


