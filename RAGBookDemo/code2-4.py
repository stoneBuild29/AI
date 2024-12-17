import torch

# test the version and function
print("PyTorch version:", torch.__version__)
#TODO print("CUDA available:", torch.cuda.is_available())

# create tensor
# create tensor from Python list
tensor1 = torch.tensor([1,2,3,4])
# create tensor from tuple
tensor2 = torch.tensor((5,6,7,8))
# create tensor from Numpy array
import numpy as np
numpy_array = np.array([9,10,11,12])
tensor3 = torch.tensor(numpy_array)
# output: tensor([1, 2, 3, 4]) tensor([5, 6, 7, 8]) tensor([ 9, 10, 11, 12])
#TODO print(tensor1, tensor2, tensor3)


# create tensor specifying the data type
tensor_float = torch.tensor([1,2,3], dtype=torch.float32)
# Place the tensor on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_gpu = torch.tensor([1,2,3], device=device)
#
tensor_grad = torch.tensor([1.1,2.2,3.3], requires_grad=True)
# output: tensor([1., 2., 3.]) tensor([1, 2, 3]) tensor([1.1000, 2.2000, 3.3000], requires_grad=True)
#TODO print(tensor_float, tensor_gpu, tensor_grad)

# 
zeros_tensor = torch.zeros(2,3)
#
ones_tensor = torch.ones(3,2)
#
uniform_tensor = torch.rand(2,2)
#
formal_tensor = torch.randn(3,3)
"""
output: tensor([[0., 0., 0.], [0., 0., 0.]]) 
        tensor([[1., 1.],[1., 1.],[1., 1.]]) 
        tensor([[0.4946, 0.4684],[0.5264, 0.1997]]) 
        tensor([[-1.1237, -0.1968,  1.3039],[-0.9188,  1.2655,  0.0761],[ 0.8051, -0.7499,  0.1268]])
"""
#TODO print(zeros_tensor, ones_tensor, uniform_tensor, formal_tensor)

# create tensor utilizing the existing tensor
tensor_original = torch.zeros(2,3)
tensor_generated1 = torch.empty_like(tensor_original)
tensor_generated2 = torch.ones_like(tensor_original)
'''
tensor([[0., 0., 0.], [0., 0., 0.]]) 
tensor([[0., 0., 0.],[0., 0., 0.]]) 
tensor([[1., 1., 1.],[1., 1., 1.]])
'''
print(tensor_original, tensor_generated1, tensor_generated2)









