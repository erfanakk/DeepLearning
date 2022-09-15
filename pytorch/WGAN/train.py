import torch
import  data_setup

import torch.nn as nn
from  model_builder import Discriminator, Critic, init_weight 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys

#gper parametr
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 5 #number of epochs #TODO
z_dim = 128
image_channel = 1
LEARNING_RATE = 5e-5
batch_size = 64
n_feature = 64
critic_iter = 5
weight_clip= 0.01




dataloader = data_setup.creat_dataset(batch_size)

dis = Critic(in_channels=1, n_feature=n_feature).to(device)
generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)
init_weight(dis)
init_weight(generat)

fixed_size = torch.randn(size=(32, z_dim, 1, 1)).to(device)

fake_writer = SummaryWriter(f'runs/DCGAN/fake')
real_writer = SummaryWriter(f'runs/DCGAN/real')
step = 0




dic_opt = torch.optim.RMSprop(dis.parameters() , lr=LEARNING_RATE)
gen_opt = torch.optim.RMSprop(generat.parameters() , lr=LEARNING_RATE)



for epoch in  range(NUM_EPOCHS):

    for batch_idx , (real, _) in enumerate(dataloader):
        dis.train()
        generat.train()
        real = real.to(device)
        for _ in range(critic_iter):

        
            noise = torch.randn(size=(batch_size,z_dim,1,1)).to(device)
            fake = generat(noise)
        
        #train dicsriminator
            
            disc_real = dis(real).reshape(-1)
            disc_fake = dis(fake).reshape(-1)

            lossD = -(torch.mean(disc_real) - torch.mean(disc_fake))

            dic_opt.zero_grad()
            lossD.backward(retain_graph=True) 
            dic_opt.step()
        
            for w in dis.parameters():
                w.data.clamp_(-weight_clip, weight_clip)


        #train generator min log(1-D(G(z)))

        output = dis(fake).reshape(-1)
        lossG = -torch.mean(output)
        gen_opt.zero_grad()
        lossG.backward()
        gen_opt.step()




        if batch_idx == 0:

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




