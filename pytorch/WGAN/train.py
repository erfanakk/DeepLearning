import torch
import torch.nn
import torchvision
from data_setup import datasetmnist
from wgan_simple import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import time
device = "cuda" if torch.cuda.is_available() else 'cpu'


epochs = 20
bc_size = 32
lr = 0.00005
z_dim = 100
n_critic = 25
img_size = 28
clip_value = 0.005
channels = 1
img_shape = (1,28,28)




gen = Generator().to(device)
critic = Critic(img_shape).to(device)


optG = torch.optim.RMSprop(gen.parameters(), lr)
optC = torch.optim.RMSprop(critic.parameters(), lr)

gen.apply(weights_init)
critic.apply(weights_init)

fake_writer = SummaryWriter(f'runs/wganSimple/fake')
step = 0
data = datasetmnist(bc_size, img_size, mnist=True, cleb=False)


def train():
    global step
    for epoch in range(epochs):
        tic = time.time()            
        if (epoch+1) % 5 == 0:
            checkpoint = {
            'gen_dict' : gen.state_dict(),
            'dic_dict' : critic.state_dict()
            }
            save_checkpoint(stete=checkpoint)

        for batch_idx , (reals, _) in enumerate(data):

            real_img = reals.to(device)
            
            #train critic
            optC.zero_grad()
            
            critic.train()
            
            z = torch.randn(size=(real_img.shape[0], z_dim)).to(device)

            fake_img = gen(z).detach( )

            lossD = - torch.mean(critic(real_img)) + torch.mean(critic(fake_img))
            
            lossD.backward()
            optC.step()
            
            
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)


            if batch_idx % n_critic == 0:

                optG.zero_grad()

                fake_img_gen = gen(z)
                lossG = - torch.mean(critic(fake_img_gen))

                lossG.backward()
                optG.step()
            
                        
            if batch_idx == (len(data)-2) :
                print(
                            f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(data)} \
                            Loss D: {lossD:.2f}, loss G: {lossG:.4f}"
                )
                toc = time.time()
                t = toc - tic
                print(f'for {epoch} wait {t:.3f}s')

                with torch.inference_mode():
                    gen.eval()
                    critic.eval()
                    fake_test = gen(z).cpu()
                    # save images in grid of 10 * 10
                    torchvision.utils.save_image(fake_test, f"mnist_epoch_{epoch+1}.jpg", nrow=10, padding=0, normalize=True)
                    img_grid = torchvision.utils.make_grid(fake_test, normalize=True)
                    fake_writer.add_image('mnist wgan fake images', img_grid, global_step=step)

                    step += 1


if __name__ == "__main__":
    train()