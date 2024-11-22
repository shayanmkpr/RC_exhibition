import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

class ReservoirComputing(nn.Module):
    def __init__(self, n_nodes, input_size, output_size , alpha_0 , gamma_0 , omega_0 , device):
        super(ReservoirComputing, self).__init__()

        self.n_nodes = n_nodes
        self.input_size = input_size
        self.output_size = output_size

        self.Wih = nn.Linear(input_size, n_nodes, bias=True).to(device)
        self.Whh = nn.Linear(n_nodes, n_nodes, bias=True).to(device)
        self.v = nn.Linear(n_nodes, 1, bias=False).to(device)
        self.Who = nn.Linear(n_nodes, output_size, bias=True).to(device)

        self.alpha = nn.Parameter(torch.normal(alpha_0, 0.1 * alpha_0, size=(n_nodes,)).to(device), requires_grad=False)
        self.gamma = nn.Parameter(torch.normal(gamma_0, 0.1 * gamma_0, size=(n_nodes,)).to(device), requires_grad=False)
        self.omega = nn.Parameter(torch.normal(omega_0, 0.1 * omega_0, size=(n_nodes,)).to(device), requires_grad=False)

    def forward(self, X, Y, h, input):
        input = torch.tensor([input]).to(device = self.alpha.device)

        I_ext = self.Wih(input).to(self.alpha.device)
        I_rec = (self.Whh(Y) + self.v(X)).to(self.alpha.device)

        # Broadcasting to apply different alpha, gamma, and omega to each node
        alpha = self.alpha.view(1, -1)
        gamma = self.gamma.view(1, -1)
        omega = self.omega.view(1, -1)

        Y_t = Y + h * (alpha * torch.tanh((1 / torch.sqrt(torch.tensor(self.n_nodes, device=self.alpha.device))) * I_rec + I_ext) - 2 * gamma * Y - omega**2 * X)
        X_t = X + h * Y_t

        return self.Who(X_t), X_t, Y_t

#testing
testing = False

if testing != False:
    device = "cpu"

    n_nodes = 16
    input_size = 1
    output_size = 10
    h = 0.1
    time_steps = 1000

    model = ReservoirComputing(n_nodes, input_size, output_size).to(device)

    # Initial states
    X = torch.zeros(n_nodes).to(device)
    Y = torch.zeros(n_nodes).to(device)
    initial_input = torch.randn(input_size).to(device)

    # Lists to store states and outputs
    X_states = []
    Y_states = []
    outputs = []

    # Simulate the model for a number of time steps
    for t in range(time_steps):
        if t == 0:
            out, X, Y = model(X, Y, h, initial_input)
        else:
            out, X, Y = model(X, Y, h, torch.zeros(input_size).to(device))
        X_states.append(X[0].cpu().detach().numpy())
        Y_states.append(Y[0].cpu().detach().numpy())
        outputs.append(out.cpu().detach().numpy())

    # Convert lists to numpy arrays for plotting
    X_states = np.array(X_states)
    Y_states = np.array(Y_states)
    outputs = np.array(outputs).squeeze()

    # Plot the changes of the states through time
    plt.figure(figsize=(8, 6))
    for i in range(n_nodes):
        plt.plot(X_states[:, i])
    plt.title('Changes of the States (X_t) Through Time')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.show()

    # Plot the changes of the states through time
    plt.figure(figsize=(8, 6))
    for i in range(n_nodes):
        plt.plot(Y_states[:, i])
    plt.title('Changes of the Y_t Through Time')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.show()

    # Plot the outputs of the model through time
    plt.figure(figsize=(8, 6))
    for i in range(output_size):
        plt.plot(outputs[:, i])
    plt.title('Outputs of the Model Through Time')
    plt.xlabel('Time Step')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()