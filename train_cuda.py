import torch
import ctypes
import numpy as np

# Load the CUDA kernel
cuda_kernel = ctypes.cdll.LoadLibrary('./custom_kernel.so')

def custom_forward(input, weights, bias, output_dim):
    input = input.contiguous()
    weights = weights.contiguous()
    bias = bias.contiguous()
    output = torch.zeros(output_dim, device='cuda', dtype=torch.float32)

    # Launch the kernel
    threads_per_block = 256
    blocks_per_grid = (output_dim + threads_per_block - 1) // threads_per_block
    cuda_kernel.custom_forward(input.data_ptr(), weights.data_ptr(), bias.data_ptr(),
                                output.data_ptr(), input.size(0), output_dim, 
                                blocks_per_grid, threads_per_block)

    return output

def custom_backward(input, weights, bias, grad_output, input_dim, output_dim):
    grad_input = torch.zeros_like(input, device='cuda')
    grad_weights = torch.zeros_like(weights, device='cuda')
    grad_bias = torch.zeros_like(bias, device='cuda')

    # Launch the kernel
    threads_per_block = 256
    blocks_per_grid = (max(input_dim, output_dim) + threads_per_block - 1) // threads_per_block
    cuda_kernel.custom_backward(input.data_ptr(), weights.data_ptr(), bias.data_ptr(), 
                                 grad_output.data_ptr(), grad_input.data_ptr(), 
                                 grad_weights.data_ptr(), grad_bias.data_ptr(), 
                                 input_dim, output_dim, blocks_per_grid, threads_per_block)

    return grad_input, grad_weights, grad_bias


# testing the cuda kernel

data = torch.tensor(..., device='cuda', dtype=torch.float32)
targets = torch.tensor(..., device='cuda', dtype=torch.float32)
input_dim = data.size(1)
output_dim = targets.size(1)
batch_size = data.size(0)

weights = torch.randn(input_dim, output_dim, device='cuda', requires_grad=True)
bias = torch.randn(output_dim, device='cuda', requires_grad=True)

learning_rate = 0.001
num_epochs = 100

for epoch in range(num_epochs):
    outputs = custom_forward(data, weights, bias, output_dim)
    loss = torch.mean((outputs - targets) ** 2)

    loss_grad = 2 * (outputs - targets) / outputs.numel()
    grad_input, grad_weights, grad_bias = custom_backward(data, weights, bias, loss_grad, input_dim, output_dim)

    with torch.no_grad():
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item()}")

torch.save({'weights': weights, 'bias': bias}, 'model.pth')
