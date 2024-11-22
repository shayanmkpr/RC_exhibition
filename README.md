# This Repository is for Exhibition Purposes Only

This repository is not intended to represent the official project. It is a showcase of the foundational experiments, testing, and initial implementations that contributed to the development of the current project. 

Please note:
- The project is still under development and not yet complete.
- The finalized code and methodologies will be made publicly available only after the official project is published.
- This repository does not include the direct methodologies or complete implementations used in the official project.

It is solely meant to provide an overview of the preliminary steps and ideas that inspired the ongoing work.


# Heterogeneous Damped Harmonic Oscillator Reservoir Computing
This repository contains the implementation of a Reservoir Computing model using Damped Harmonic Oscillator (DHO) nodes for processing sequential data. The RC-HORNN model is designed to efficiently handle temporal information with dynamic reservoir states and supports training via both **Backpropagation Through Time (BPTT)** and **Hebbian Learning**.

## Model Description

The RC-HORNN consists of:
- **Reservoir**: A collection of DHO nodes with heterogeneous characteristics. Each node's dynamics are defined by the following equations:

    $$I_{\text{rec}}(t) = W_{\text{hh}} y(t) + b_{\text{hh}} + v \cdot x(t)$$

    $$I_{\text{ext}}(t) = W_{\text{ih}} s(t) + b_{\text{ih}}$$

    $$x_{t+1} = x_t + h y_{t+1}$$

    $$y_{t+1} = y_t + h \left[ \alpha \cdot \tanh\left(\frac{1}{\sqrt{n}} I_{\text{rec}}(t) + I_{\text{ext}}(t)\right) - 2\gamma \cdot y_t - \omega^2 \cdot x_t \right]$$

    - \(x_t\): Node states
    - \(y_t\): Derivative of node states
    - \(\alpha, \gamma, \omega\): Node-specific parameters for dynamics

- **Input Layer**: Provides weighted connections to each node in the reservoir.
- **Output Layer**: Maps the reservoir states to the output predictions.

## Training Methods

### 1. Backpropagation Through Time (BPTT)
- **Description**: The input and output layers are trained using BPTT, while the reservoir weights remain fixed.
- **Process**:
  1. Sequential data is fed to the model step by step.
  2. Cross-entropy loss is computed between the predicted and true labels.
  3. Gradients are propagated back through the time steps to update input-to-hidden (\(W_{\text{ih}}\)) and hidden-to-output (\(W_{\text{ho}}\)) weights.

### 2. Hebbian Learning with BPTT
- **Description**: In addition to BPTT for the input and output layers, the reservoir's hidden weights (\(W_{\text{hh}}\)) are updated using Hebbian learning.
- **Hebbian Rule**:
    $$\Delta W_{ij} = \sigma \lambda h a_{ij} r(x_i(t), x_j(t))$$
    
    Where:
    - \(r(x_i, x_j)\): Pearson correlation between node states \(x_i\) and \(x_j\)
    - \(a_{ij}\): Active connection condition based on thresholds
    - \(\sigma\): Scaling factor for Hebbian learning
