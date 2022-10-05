from cgan import Discriminator , Generator, load_checkpoint
import torch
import torch.nn as nn
import numpy as np
import torchvision


'''
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

'''
device = 'cpu'

z_dim = 100
bc_size = 32
num_class_gen = [0,1,2,3,4,5,6,7,8,9]

onehot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)

fill = torch.zeros([10, 10, 28, 28])
for i in range(10):
    fill[i, i, :, :] = 1


label_real = torch.ones((bc_size,1)).to(device)
label_fake = torch.zeros((bc_size,1)).to(device)
z_test = torch.randn(size=(100, z_dim)).to(device)




gen = Generator().to(device)
dic = Discriminator().to(device)
load_checkpoint(torch.load('CGANs.pth.tar') , dic=dic, gen=gen)

gen.eval()

for num in num_class_gen:
    test_y = torch.tensor([num]*100).type(torch.LongTensor)
    test_Gy = onehot[test_y].to(device)        

    fake_test = gen(z_test, test_Gy).cpu()
    # save images in grid of 10 * 10
    torchvision.utils.save_image(fake_test, f"mnist_{num}_.jpg", nrow=10, padding=0, normalize=True)

