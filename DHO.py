import torch
import torch.nn as nn

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
        # input = torch.tensor(input).to(device = self.alpha.device)

        I_ext = self.Wih(input).to(self.alpha.device)
        I_rec = (self.Whh(Y) + self.v(X)).to(self.alpha.device)

        alpha = self.alpha.view(1, -1)
        gamma = self.gamma.view(1, -1)
        omega = self.omega.view(1, -1)

        Y_t = Y + h * (alpha * torch.tanh((1 / torch.sqrt(torch.tensor(self.n_nodes, device=self.alpha.device))) * I_rec + I_ext) - 2 * gamma * Y - omega**2 * X)
        X_t = X + h * Y_t

        return self.Who(X_t), X_t, Y_t