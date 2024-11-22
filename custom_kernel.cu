#include <cuda_runtime.h>

extern "C" {
    __global__ void custom_forward(
        float* d_X, float* d_Y, const float* d_input, const float* d_Wih, 
        const float* d_Whh, const float* d_v, const float* d_Who, 
        const float* d_alpha, const float* d_gamma, const float* d_omega, 
        float* d_output, int n_nodes, int input_size, float h) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_nodes) return;

        float I_ext = 0.0f;
        for (int j = 0; j < input_size; j++) {
            I_ext += d_Wih[idx * input_size + j] * d_input[j];
        }

        float I_rec = 0.0f;
        for (int j = 0; j < n_nodes; j++) {
            I_rec += d_Whh[idx * n_nodes + j] * d_Y[j];
        }
        I_rec += d_v[idx] * d_X[idx];

        float tanh_input = (1.0f / sqrtf(n_nodes)) * (I_rec + I_ext);
        float tanh_value = tanhf(tanh_input);

        float alpha = d_alpha[idx];
        float gamma = d_gamma[idx];
        float omega = d_omega[idx];

        float Y_t = d_Y[idx] + h * (alpha * tanh_value - 2 * gamma * d_Y[idx] - (omega * omega) * d_X[idx]);
        float X_t = d_X[idx] + h * Y_t;

        d_output[idx] = d_Who[idx] * X_t;

        d_X[idx] = X_t;
        d_Y[idx] = Y_t;
}
}

extern "C" __global__ void custom_backward(float* input, float* weights, float* bias, float* grad_output, 
                                           float* grad_input, float* grad_weights, float* grad_bias, 
                                           int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_dim) {
        atomicAdd(&grad_bias[idx], grad_output[idx]);
    }

    for (int i = 0; i < input_dim; ++i) {
        if (idx < output_dim) {
            atomicAdd(&grad_weights[i * output_dim + idx], input[i] * grad_output[idx]);
        }
        if (idx < input_dim) {
            atomicAdd(&grad_input[i], grad_output[idx] * weights[i * output_dim + idx]);
        }
    }
}
