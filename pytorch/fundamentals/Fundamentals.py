# import torch
# from torch import nn
# import matplotlib.pyplot as plt
# import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"


#-----------------------#

#tensor in torch 
# ts = torch.tensor(7)

# #remove tensor 
# tsITEM = ts.item() 

# #dim of tensor
# tsDIM = ts.ndim
# print('this is shape of tensor' , ts.shape) 

#------------------------#

#vector in torch 
# vector = torch.tensor([5,4,2])
# vectorDIM = vector.ndim
# vectorShape = vector.shape
# print('this is the vector' , vector)
# print('dim of vector' , vectorDIM)
# print('shape of vector' , vectorShape)


#------------------------#

#Matrix in torch

# Matrix = torch.tensor([[5,4,2] , [1,2,4]])
# MDIM = Matrix.ndim
# MShape = Matrix.shape
# print('this is the matric' , Matrix)
# print('dim of matrix' , MDIM)
# print('shape of matrix' , MShape)

#------------------------#

#random tensor 

# create a random tensor of size (2,4)
# randomTensor = torch.rand(size=(2,4))
# print(randomTensor)

#random image size
# random_image_size_tensor = torch.rand(size=(224, 224, 3))
# print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

#------------------------#

#create tensor zeros and ones

# zeros = torch.zeros(size=(2,4))
# ones = torch.ones(size=(2,4))

# print('this is ones' , ones)
# print('this is zeros' , zeros)

#------------------------#

#range in torch 
# zero_to_ten = torch.arange(start=0, end=10, step=1)
# print(zero_to_ten)

# Can also create a tensor of zeros and ones similar to another tensor
# tenZeros = torch.zeros_like(zero_to_ten)
# print(tenZeros)
# tenOnes = torch.ones_like(zero_to_ten)
# print(tenOnes)

#------------------------#

#data type in torch commen torch.float32 

# float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
#                                dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
#                                device=None, # defaults to None, which uses the default tensor type
#                                requires_grad=False) # if True, operations perfromed on the tensor are recorded 

# print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

#------------------------#

# tensor = torch.tensor([[3,4,87],[45,99,34]])

# print('this is element wise')
# print(tensor * tensor )
# print('this matrix mul')
# print(torch.matmul(tensor, tensor))
# print(tensor)
# print(tensor.T)
# print(tensor[0][2].item())
# print(tensor.amax())
# print(tensor.argmax())

#------------------------#

#Reshaping, stacking, squeezing and unsqueezing
'''
Method	One-line description
torch.reshape(input, shape)   	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
torch.Tensor.view(shape)	    Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
torch.stack(tensors, dim=0)     	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
torch.squeeze(input)	         Squeezes input to remove all the dimenions with value 1.
torch.unsqueeze(input, dim)	     Returns input with a dimension value of 1 added at dim.
torch.permute(input, dims)    	Returns a view of the original input with its dimensions permuted (rearranged) to dims.

'''

# x = torch.arange(1., 8.)
# print(x, x.shape)

# x_reshape = x.reshape(1,7)
#print(x_reshape, x_reshape.shape)

# Change view (keeps same data as original but changes view)
# z = x.view(1,7)
# print(z, z.shape)


# print(f"Previous tensor: {x_reshape}")
# print(f"Previous shape: {x_reshape.shape}")

# # Remove extra dimension from x_reshaped

# x_squeezed = x_reshape.squeeze()
# print(f"\nNew tensor: {x_squeezed}")
# print(f"New shape: {x_squeezed.shape}")


# Create tensor with specific shape
# x_original = torch.rand(size=(224, 224, 3))

# # Permute the original tensor to rearrange the axis order
# x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

# print(f"Previous shape: {x_original.shape}")
# print(f"New shape: {x_permuted.shape}")

#------------------------#

# array = np.arange(1.0, 8.0)
# tensor = torch.from_numpy(array).type(torch.float32)
# print(array, tensor)

# tensor = torch.ones(8)
# array = tensor.numpy()
# print(array, tensor)

#------------------------#


# import random
# # # Set the random seed
# RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
# torch.manual_seed(seed=RANDOM_SEED) 
# random_tensor_C = torch.rand(3, 4)

# # Have to reset the seed every time a new rand() is called 
# # Without this, tensor_D would be different to tensor_C 
# torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
# random_tensor_D = torch.rand(3, 4)

# print(f"Tensor C:\n{random_tensor_C}\n")
# print(f"Tensor D:\n{random_tensor_D}\n")
# print(f"Does Tensor C equal Tensor D? (anywhere)")
# print(random_tensor_C == random_tensor_D)


#------------------------#

#loss function
'''
Stochastic Gradient Descent (SGD) optimizer	         Classification, regression, many others.	    torch.optim.SGD()
Adam Optimizer       	Classification, regression, many others.	        torch.optim.Adam()
Binary cross entropy loss	Binary classification	torch.nn.BCELossWithLogits or torch.nn.BCELoss
Cross entropy loss	Mutli-class classification	torch.nn.CrossEntropyLoss
Mean absolute error    (MAE) or L1 Loss	   Regression	torch.nn.L1Loss
Mean squared error (MSE) or L2 Loss	       Regression	     torch.nn.MSELoss

'''



