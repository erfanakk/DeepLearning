import torch
import  data_setup
import utils
import torch.nn as nn
from  model_builder import Discriminator, Critic, init_weight 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys
import argparse
#gper parametr
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"



parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

opt = parser.parse_args()


NUM_EPOCHS = opt.epoch #number of epochs #TODO
z_dim = opt.latent_dim
image_channel = opt.channels
LEARNING_RATE = opt.lr
batch_size = opt.batch_size
n_feature = 64
critic_iter = opt.n_critic
weight_clip= opt.clip_value












dataloader = data_setup.creat_dataset(batch_size)

dis = Critic(in_channels=1, n_feature=n_feature).to(device)
generat = Generator(z_dim=z_dim, n_feature=n_feature, img_chan=image_channel).to(device)
init_weight(dis)
init_weight(generat)

fixed_size = torch.randn(size=(32, z_dim, 1, 1)).to(device)

fake_writer = SummaryWriter(f'runs/WGAN/fake')
real_writer = SummaryWriter(f'runs/WGAN/real')
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

    if save_model:
        if epoch % 5 == 0:
            checkpoint = {
                        'state_dic_GEN' : generat.state_dict(),
                        'state_dic_DIS' : dis.state_dict()
                        }
            utils.save_checkpoint(stete=checkpoint) 


