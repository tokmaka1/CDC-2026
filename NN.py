# Bayesian scenario approach
# Create functions from a probability disatribution
# Here we try it on NN parameters; this can be a simple NN, a Bayesian NN
# Preferably this is a Bayesian NN, then we have posterior updates
# We have already done this in Hertneck et al. with Hoeffding; a posteriori validation of MPC with indicator functions
# Tokmak et al. 2024 with Hoeffding; Tokmak et al. 2025 with scenario approach
# Here we do not do validation but in-loop error bounds...


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 


# Define the network structure
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 10)   # 1 input → 10 hidden
        self.output = nn.Linear(10, 1)   # 10 hidden → 1 output

    def forward(self, x):
        x = torch.tanh(self.hidden(x))  # Activation
        return self.output(x)           # Linear output

    def set_weights(self, theta):
        # Assume theta is a flat tensor of length 31
        with torch.no_grad():
            idx = 0
            # Hidden layer weights
            W1_shape = self.hidden.weight.shape  # (10,1)
            b1_shape = self.hidden.bias.shape    # (10,)
            W1_num = np.prod(W1_shape)
            b1_num = np.prod(b1_shape)

            self.hidden.weight.copy_(theta[idx:idx+W1_num].view(W1_shape))
            idx += W1_num
            self.hidden.bias.copy_(theta[idx:idx+b1_num].view(b1_shape))
            idx += b1_num

            # Output layer weights
            W2_shape = self.output.weight.shape  # (1,10)
            b2_shape = self.output.bias.shape    # (1,)
            W2_num = np.prod(W2_shape)
            b2_num = np.prod(b2_shape)

            self.output.weight.copy_(theta[idx:idx+W2_num].view(W2_shape))
            idx += W2_num
            self.output.bias.copy_(theta[idx:idx+b2_num].view(b2_shape))

# Do the RKHS business here! Directly with pre-RKHS functions...
# We can say these are RKHS functions with randomly sampled alpha; keep the x as we have also in the RKHS paper; RKHS does not imply frequentist
# Make the RKHS

if __name__ == '__main__':
    # Total number of parameters
    net = SimpleNN()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params}")  # Should be 31

    # Sample random weights and evaluate
    plt.figure()
    for _ in range(10):
        theta = torch.empty(total_params).uniform_(-1, 1)
        net.set_weights(theta)
        x_vals = torch.linspace(0, 1, 500).view(-1, 1)
        with torch.no_grad():
            y_vals = net(x_vals).squeeze().numpy()
        plt.plot(x_vals.numpy(), y_vals)

