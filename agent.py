import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from dueling_model import DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

USE_DOUBLE_DQN = False  # whether or not to use double dqn
USE_PRIORITIZED_REPLAY = False  # use prioritized experience replay
USE_DUELING_NETWORK = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ An agent to interact with the environment and learn from it """
    
    def __init__(self,state_size, action_size, seed):
        """ Initialization function. 
        
        Params
        ======
            state_size (int): dim of each state
            action_size (int): dim of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # define Q-Network
        if USE_DUELING_NETWORK:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed,128,32,64,32).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed,128,32,64,32).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory
        if USE_PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed,device, alpha=.6, beta = .4, beta_scheduler=1.)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # initial time step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # save experience in memory replay
        self.memory.add(state, action, reward, next_state, done)
        # learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step  == 0:
            # when the memory is full enough
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0):
        """ Return actions for given state as per current policy
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # epsilon-greedy action selection 
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.
        
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, w) tuples
            gamma (float): discount factor
        """
        
        if USE_PRIORITIZED_REPLAY:
            states, actions, reward, next_states, dones, w = experiences
        else:
            states, actions, reward, next_states, dones = experiences
            
        with torch.no_grad():
            if USE_DOUBLE_DQN :
                Q_local_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                Q_target_next = self.qnetwork_target(next_states).gather(1,Q_local_next)
                
            else:                
                # get max predicted Q values (for next states) from target model
                Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # compute Q targets for current states
            Q_target = reward + (gamma * Q_target_next * (1 - dones))

        # get expected Q values from local model
        Q_expected  = self.qnetwork_local(states).gather(1,actions)
        
        if USE_PRIORITIZED_REPLAY:
            Q_target.sub_(Q_expected)
            Q_target = torch.squeeze(Q_target)
            Q_target.pow_(2)
            with torch.no_grad():
                TD_error = Q_target.detach()
                TD_error.pow_(.5)
                self.memory.update_priorities(TD_error)
            
            Q_target.mul_(w)
            loss = Q_target.mean()
        else:
            # compute loss
            loss = F.mse_loss(Q_expected, Q_target)
            
        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
    
    def soft_update(self, local_model, target_model, tau):
        """ soft update model parameters.
        theta_target = tau*theta_local + (1 - tau)*theta_target
        
        Params
        ======
            local_model (pytorch model): weight will be copied from
            target-model (pytorch model): weight will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1. -tau)*target_param.data)
            
            
            
class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dim of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """ add a new experience to memory """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """ randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """ return the current size of the internal memory. """
        return len(self.memory)
        
class PrioritizedReplayBuffer:
    """Fixed-size prioritized buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha=0., beta=1., beta_scheduler=1.):
        """ initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dim of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): prioritized lever alpha=0 -> uniform case
            beta (float): level of importance-sampling corretion, beta=1 -> compensates for the non-uniform proba
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_scheduler = beta_scheduler
        
        # create memory
        self.memory = np.empty(buffer_size, dtype=[
            ("state", np.ndarray),
            ("action", np.int),
            ("reward", np.float),
            ("next_state", np.ndarray),
            ("done", np.bool),
            ('prob', np.double)])
        
        self.memory_circular = 0
        
        self.memory_samples_indices = np.empty(self.batch_size)
        self.memory_samples = np.empty(self.batch_size, dtype=type(self.memory))
        
        self.max_prob = 0.0001
        self.nonzero_probability = 0.00001
        
        self.p = np.empty(self.buffer_size, dtype=np.double)
        self.w = np.empty(self.buffer_size, dtype=np.double)
        
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        # Add the experienced parameters to the memory
        self.memory[self.memory_circular]['state'] = state
        self.memory[self.memory_circular]['action'] = action
        self.memory[self.memory_circular]['reward'] = reward
        self.memory[self.memory_circular]['next_state'] = next_state
        self.memory[self.memory_circular]['done'] = done
        self.memory[self.memory_circular]['prob'] = self.max_prob
        
        # Control memory as a circular list
        self.memory_circular = (self.memory_circular + 1) % self.buffer_size
    
    
    def sample(self):
        """Sample a batch of prioritized experiences from memory."""
        
        # Normalize the probability of being chosen for each one of the memory registers
        np.divide(self.memory['prob'], self.memory['prob'].sum(), out=self.p)
        # Choose "batch_size" sample index following the defined probability
        self.memory_samples_indices = np.random.choice(self.buffer_size, self.batch_size, replace=False, p=self.p)
        # Get the samples from memory
        self.memory_samples = self.memory[self.memory_samples_indices]
        
        # Compute importance-sampling weights for each one of the memory registers
        # w = ((N * P) ^ -β) / max(w)
        np.multiply(self.memory['prob'], self.buffer_size, out=self.w)
        np.power(self.w, -self.beta, out=self.w, where=self.w!=0) # condition to avoid division by zero
        np.divide(self.w, self.w.max(), out=self.w) # normalize the weights
        
        self.beta = min(1, self.beta*self.beta_scheduler)
        
        # Split data into new variables
        states = torch.from_numpy(np.vstack(self.memory_samples['state'])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(self.memory_samples['action'])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(self.memory_samples['reward'])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(self.memory_samples['next_state'])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(self.memory_samples['done']).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(self.w[self.memory_samples_indices]).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, weights)
    
    def update_priorities(self, td_error):
        # Balance the prioritization using the alpha value
        td_error.pow_(self.alpha)

        # Guarantee a non-zero probability
        td_error.add_(self.nonzero_probability)
        
        # Update the probabilities in memory
        self.memory_samples['prob'] = td_error
        self.memory[self.memory_samples_indices] = self.memory_samples
        
        # Update the maximum probability value
        self.max_prob = self.memory['prob'].max()
        
       
    def __len__(self):
        """Return the current size of internal memory."""
        return self.buffer_size if self.memory_circular // self.buffer_size > 0 else self.memory_circular