import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class RIALAgent:
    def __init__(self, state_dim, action_dim, comm_dim=1, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.comm_dim = comm_dim  # 1-bit communication
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Main network (action + communication)
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=10000)
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim + self.comm_dim, 64),  # State + received message
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim + self.comm_dim)  # Output: action + message
        )
        return model
    
    def act(self, state, received_message=None):
        if received_message is None:
            received_message = np.zeros(self.comm_dim)
        
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_dim)
            message = np.random.choice(2)  # Binary communication
            return action, message
        
        state_with_msg = np.concatenate([state, received_message])
        state_with_msg = torch.FloatTensor(state_with_msg).unsqueeze(0)
        output = self.model(state_with_msg)
        
        action = torch.argmax(output[:, :self.action_dim]).item()
        message = torch.sigmoid(output[:, -self.comm_dim]).round().item()  # Binary message
        
        return action, message
    
    def remember(self, state, action, message, reward, next_state, done):
        self.memory.append((state, action, message, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, messages, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        messages = torch.FloatTensor(messages)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Combine state and message
        states_with_msg = torch.cat([states, messages.unsqueeze(1)], dim=1)
        next_states_with_msg = torch.cat([next_states, torch.zeros_like(messages).unsqueeze(1)], dim=1)  # Placeholder for next message
        
        current_q = self.model(states_with_msg)
        next_q = self.model(next_states_with_msg).max(1)[0].detach()
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q[:, :self.action_dim].gather(1, actions.unsqueeze(1)).squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))