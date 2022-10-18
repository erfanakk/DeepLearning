import numpy as np
import time

## numpy


# a = np.random.random_sample(4)
# print(a.shape)
# a = np.zeros(shape=(5,))
# print(a)
# a = np.arange(4)
# print(a)
# a = np.random.rand(4)
# print(a)
# a = np.random.uniform(high=5.0, low=1.0,size=(5))
# print(a)
# print(np.all(a > 0))
# a = np.array([1,5,3,5,5,3,1,3])
# print(np.unique(a))

# #vector slicing operations
# a = np.arange(10)
# print(f"a         = {a}")

# #access 5 consecutive elements (start:stop:step)
# c = a[2:7:1];     print("a[2:7:1] = ", c)
# a = np.array([1,2,3,4])
# b = -a 
# b = np.sum(a) 
# b = np.mean(a)
# b = a**2

# def my_dot(a, b): 
#     """
#    Compute the dot product of two vectors
 
#     Args:
#       a (ndarray (n,)):  input vector 
#       b (ndarray (n,)):  input vector with same dimension as a
    
#     Returns:
#       x (scalar): 
#     """
#     x=0
#     for i in range(a.shape[0]):
#         x = x + a[i] * b[i]
#     return x



# np.random.seed(1)
# a = np.random.rand(10000000)  # very large arrays
# b = np.random.rand(10000000)

# tic = time.time()  # capture start time
# c = np.dot(a, b)
# toc = time.time()  # capture end time

# print(f"np.dot(a, b) =  {c:.4f}")
# print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

# tic = time.time()  # capture start time
# c = my_dot(a,b)
# toc = time.time()  # capture end time

# print(f"my_dot(a, b) =  {c:.4f}")
# print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

# del(a)
# del(b)  #remove these big arrays from memory

