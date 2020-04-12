import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_adv=64, fc2_adv=64,fc1_val=64,fc2_val=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_adv (int): Number of nodes in first advantage hidden layer
            fc2_adv (int): Number of nodes in second advantage hidden layer
            fc1_val (int): Number of nodes in first value hidden layer
            fc2_val (int): Number of nodes in second value hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1_advantage = nn.Linear(state_size, fc1_adv)
        self.fc2_advantage = nn.Linear(fc1_adv, fc2_adv)
        self.fc3_advantage = nn.Linear(fc2_adv, action_size)
        
        self.fc1_value = nn.Linear(state_size, fc1_val)
        self.fc2_value = nn.Linear(fc1_val, fc2_val)
        self.fc3_value = nn.Linear(fc2_val, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x1 = F.relu(self.fc1_advantage(state))
        x1 = F.relu(self.fc2_advantage(x1))
        x1 = self.fc3_advantage(x1)
        
        x2 = F.relu(self.fc1_value(state))
        x2 = F.relu(self.fc2_value(x2))
        x2 = self.fc3_value(x2)
        
        y = x1.sub_(x1.mean()).add_(x2)
        
        return y
