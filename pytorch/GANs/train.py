import torch
import data_setup
import torch.nn as nn
from  model_builder import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys

#gper parametr
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 1 #TODO
z_dim = 64
image_dim = 28 * 28 * 1
LEARNING_RATE = 3e-4
batch_size = 32


dataloader = data_setup.creat_dataset(batch_size)

dis = Discriminator(in_channels=image_dim).to(device)
generat = Generator(z_dim=z_dim, img_dim=image_dim).to(device)
fixed_size = torch.rand(size=(batch_size,z_dim)).to(device)
fake_writer = SummaryWriter(f'runs/Gan_mnist/fake')
real_writer = SummaryWriter(f'runs/Gan_mnist/real')
step = 0



loss_fn = nn.BCELoss()
dic_opt = torch.optim.Adam(dis.parameters() , lr=LEARNING_RATE)
gen_opt = torch.optim.Adam(generat.parameters() , lr=LEARNING_RATE)
 

for epoch in  range(NUM_EPOCHS):
    for batch_idx , (real, _) in enumerate(dataloader):


        real = real.reshape(real.shape[0], image_dim).to(device)
        batch_size = real.shape[0]


        # train disc --> max log(D(real)) + log(1 - D(G(z)))
        noise = torch.rand(size=(batch_size,z_dim)).to(device)
        fake = generat(noise)
        dic_real = dis(real).view(-1)
        
        
        dicR_real = loss_fn(dic_real, torch.ones_like(dic_real).to(device))
        dic_fake = dis(fake).view(-1)
        dicR_fake = loss_fn(dic_fake, torch.zeros_like(dic_fake).to(device))

        lossD = (dicR_fake + dicR_real) / 2
        dic_opt.zero_grad()
        lossD.backward( retain_graph=True )
        dic_opt.step()

        #train genarator max log D(G(z))

        output = dis(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output).to(device))

        gen_opt.zero_grad()
        lossG.backward()
        gen_opt.step()
        
        if batch_idx == 0:
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




