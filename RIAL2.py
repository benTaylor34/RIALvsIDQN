import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
from pathlib import Path


#Soft attention communication module
class AttentionModule(nn.Module):
    def __init__(self, obs_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs_i, obs_j):
        x = torch.cat([obs_i, obs_j], dim=-1)
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score

#Calculate and return the weighted communication information
def compute_attention_weighted_messages(obs_dict, agents, attention_model):
    obs_tensor = {agent: torch.FloatTensor(obs).unsqueeze(0) for agent, obs in obs_dict.items()}
    messages = {}
    attention_weights_dict = {}  #Store the attention weights for subsequent training

    for agent in agents:
        scores = []
        neighbor_obs = []
        neighbors = []
        
        for other in agents:
            if other == agent:
                continue
            score = attention_model(obs_tensor[agent], obs_tensor[other])
            scores.append(score)
            neighbor_obs.append(obs_tensor[other])
            neighbors.append(other)

        if len(scores) == 0:
            messages[agent] = torch.zeros_like(obs_tensor[agent]).squeeze(0).numpy()
            attention_weights_dict[agent] = {}
            continue

        scores = torch.cat(scores, dim=0)
        attn_weights = torch.softmax(scores, dim=0)
        
        #Save the attention weights for training
        attention_weights_dict[agent] = {neighbors[i]: attn_weights[i].item() for i in range(len(neighbors))}
        
        neighbor_obs = torch.cat(neighbor_obs, dim=0)
        weighted_sum = (attn_weights * neighbor_obs).sum(dim=0)
        messages[agent] = weighted_sum.detach().numpy()

    return messages, attention_weights_dict


#Define the RIAL network structure
class ImprovedRIALNet(nn.Module):
    def __init__(self, input_dim, message_dim, agent_id_dim, hidden_dim, action_dim):
        super().__init__()
        #Simplify the structure: Directly concatenate the observations, messages and ids for processing
        total_input_dim = input_dim + message_dim + agent_id_dim
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)
        
        #Use a smaller RNN as an auxiliary rather than the main processing unit
        self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True)
        
        #The additional processing layer combines the RNN output and the FC output
        self.fc_combine = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs_input, agent_id_input, message_input, hidden):
        #Concatenate all inputs
        combined_input = torch.cat([obs_input, agent_id_input, message_input], dim=-1)
        
        #Feedforward processing
        x = F.relu(self.fc1(combined_input))
        fc_out = F.relu(self.fc2(x))
        
        #RNN processes sequence information
        rnn_in = fc_out.unsqueeze(1)  # 添加时间维度
        rnn_out, h_out = self.rnn(rnn_in, hidden)
        rnn_out = rnn_out.squeeze(1)
        
        #Merge the RNN output and the FC output
        combined = torch.cat([fc_out, rnn_out], dim=-1)
        final = F.relu(self.fc_combine(combined))
        
        #Q value output
        q_values = self.q_head(final)
        
        return q_values, h_out

#Define the RIAL agent
class ImprovedRIALAgent:
    def __init__(self, obs_dim, action_dim, num_agents,
                 hidden_dim=64, lr=0.0005, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, memory_size=50000, batch_size=64,
                 tau=0.005):  #Add soft update parameters
        
        self.input_dim = obs_dim
        self.message_dim = obs_dim
        self.agent_id_dim = num_agents
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.tau = tau  #Soft update parameter
        
        #Use the improved network
        self.model = ImprovedRIALNet(self.input_dim, self.message_dim, self.agent_id_dim, hidden_dim, action_dim)
        self.target_model = ImprovedRIALNet(self.input_dim, self.message_dim, self.agent_id_dim, hidden_dim, action_dim)
        
        #Use a lower learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.HuberLoss()  #Use Huber loss to improve stability
        
        #Use priority experience replay
        self.memory = self._setup_prioritized_replay(memory_size)
        
        #Hidden state
        self.hidden_states = {}
        self.reset_hidden()
        
        #Initialize the target network
        self.update_target_network(tau=1.0)  
    
    def _setup_prioritized_replay(self, memory_size):
        return deque(maxlen=memory_size)
    
    def reset_hidden(self):
        self.hidden_states = {i: torch.zeros(1, 1, self.hidden_dim // 2) for i in range(self.num_agents)}
    
    def act(self, obs, agent_index, message):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        agent_id_tensor = torch.eye(self.num_agents)[agent_index].unsqueeze(0)
        message_tensor = torch.FloatTensor(message).unsqueeze(0)
        
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values, self.hidden_states[agent_index] = self.model(
                obs_tensor, agent_id_tensor, message_tensor, self.hidden_states[agent_index]
            )
        return torch.argmax(q_values).item()
    
    def step(self, obs, agent_index, action, reward, next_obs, done, message):
        #Store experience
        self.memory.append((obs, agent_index, action, reward, next_obs, done, message))
        
        #Learn when sufficient samples are accumulated
        if len(self.memory) >= self.batch_size:
            self.learn()
    
    def learn(self):
        #Random sampling batch
        batch = random.sample(self.memory, self.batch_size)
        obs_batch, agent_indices, actions, rewards, next_obs_batch, dones, messages = zip(*batch)
        
        #Convert to a tensor
        obs_batch = torch.FloatTensor(np.array(obs_batch))
        agent_indices = torch.LongTensor(agent_indices)
        agent_id_batch = torch.eye(self.num_agents)[agent_indices]
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        dones = torch.FloatTensor([float(d) for d in dones])
        messages = torch.FloatTensor(np.array(messages))
        
        #Create a batch hidden state
        hidden = torch.zeros(1, self.batch_size, self.hidden_dim // 2)
        
        #Calculate the current Q value
        current_q, _ = self.model(obs_batch, agent_id_batch, messages, hidden)
        current_q = current_q.gather(1, actions).squeeze()
        
        #Calculate the Q value of the next state without gradient
        with torch.no_grad():
            #Double DQN: Select actions using the online network and evaluate the target network
            next_q_online, _ = self.model(next_obs_batch, agent_id_batch, messages, hidden)
            next_actions = next_q_online.max(1)[1].unsqueeze(1)
            
            next_q_target, _ = self.target_model(next_obs_batch, agent_id_batch, messages, hidden)
            next_q = next_q_target.gather(1, next_actions).squeeze()
            
            #Calculate the target Q value
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        #Calculate the losses and optimize them
        loss = self.loss_fn(current_q, target_q)
        
        #Optimizer steps
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradient cropping is used to improve the stability of training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        #Attenuation exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self, tau=None):
        #Soft update the target network
        tau = tau if tau is not None else self.tau
        
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

#Improve the training of the attention mechanism
class AttentionTrainer:
    def __init__(self, attention_model, lr=0.0001):
        self.model = attention_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def train_step(self, obs_dict, attention_weights, rewards_dict):
        #Train the attention model based on rewards and attention weights
        if not attention_weights:
            return 0.0
            
        loss = 0.0
        update_count = 0
        
        for agent, weights in attention_weights.items():
            if not weights:
                continue
                
            obs_i = torch.FloatTensor(obs_dict[agent]).unsqueeze(0)
            reward_i = rewards_dict[agent]
            
            #Adjust the ideal attention weight according to the rewards
            target_weights = {}
            for other, weight in weights.items():
                #Adjust the weight: If the reward is positive, enhance attention; If it is negative, it weakens attention
                reward_factor = max(0.1, min(2.0, 1.0 + 0.1 * reward_i))
                target_weights[other] = min(1.0, weight * reward_factor)
                
            #Normalize the target weight
            total = sum(target_weights.values()) + 1e-10
            for other in target_weights:
                target_weights[other] /= total
                
            #Calculate and update the gradient
            for other, target in target_weights.items():
                obs_j = torch.FloatTensor(obs_dict[other]).unsqueeze(0)
                score = self.model(obs_i, obs_j)
                
                #Create a separate target weight tensor
                target_tensor = torch.tensor([target], dtype=torch.float32)
                
                #Use sigmoid to map the score to the interval of (0,1) as the prediction weight
                pred_weight = torch.sigmoid(score)
                
                #Calculate the loss
                step_loss = self.loss_fn(pred_weight, target_tensor)
                loss += step_loss
                update_count += 1
        
        if update_count > 0:
            #Average the loss and backpropagation
            loss = loss / update_count
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
        return loss.item() if update_count > 0 else 0.0

#Training cycle
def train_improved_rial(env, shared_agent, agent_mapping, attention_model, 
                        attention_trainer, num_episodes=3000):
    reward_log = []
    loss_log = []
    attention_loss_log = []
    
    #print("Beging！！！！")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        shared_agent.reset_hidden()
        
        done = {a: False for a in env.agents}
        ep_rewards = {a: 0.0 for a in env.agents}
        step_count = 0
        episode_loss = 0.0
        attention_loss = 0.0
        
        #Each round of the game loop
        while not all(done.values()) and step_count < 100:  #Add a maximum step limit
            #Calculate the message and attention weights
            messages, attention_weights = compute_attention_weighted_messages(
                obs, env.agents, attention_model)
            
            #The agent selects the action
            actions = {}
            for agent in env.agents:
                if not done[agent]:
                    agent_idx = agent_mapping[agent]
                    actions[agent] = shared_agent.act(obs[agent], agent_idx, messages[agent])
            
            #Perform the action
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            
            #Update the attention model
            if episode % 5 == 0:  
                step_attn_loss = attention_trainer.train_step(obs, attention_weights, rewards)
                attention_loss += step_attn_loss
            
            #Each agent learns
            for agent in env.agents:
                if not done[agent]:
                    agent_idx = agent_mapping[agent]
                    shared_agent.step(
                        obs[agent], agent_idx, actions[agent], rewards[agent],
                        next_obs[agent], terms[agent] or truncs[agent], messages[agent]
                    )
                    ep_rewards[agent] += rewards[agent]
            
            #Update observations and status
            obs = next_obs
            done = {a: terms[a] or truncs[a] for a in env.agents}
            step_count += 1
        
        #Soft update the target network
        if episode % 5 == 0:  
            shared_agent.update_target_network()
        
        #Record rewards and losses
        total_reward = sum(ep_rewards.values())
        reward_log.append(total_reward)
        loss_log.append(episode_loss / max(1, step_count))
        attention_loss_log.append(attention_loss / max(1, step_count))
        
        #Output the training progress (once every 100 steps)
        if episode % 100 == 0 or episode == num_episodes - 1:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {shared_agent.epsilon:.4f}")

    return reward_log, loss_log, attention_loss_log


#Definition of Practical functions
#Calculate the moving average
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#Draw a comparison chart of the training results
def plot_training_results(dqn_rewards=None, rial_rewards=None, window_size=50):
    plt.figure(figsize=(12, 6))
    
    if dqn_rewards is not None:
        plt.plot(dqn_rewards, alpha=0.3, color='blue', label='DQN Raw')
        plt.plot(range(window_size-1, len(dqn_rewards)), 
                 moving_average(dqn_rewards, window_size), 
                 color='blue', linewidth=2, label='DQN Moving Avg')
    
    if rial_rewards is not None:
        plt.plot(rial_rewards, alpha=0.3, color='red', label='RIAL Raw')
        plt.plot(range(window_size-1, len(rial_rewards)), 
                 moving_average(rial_rewards, window_size), 
                 color='red', linewidth=2, label='RIAL Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt



if __name__ == "__main__":
    env = simple_spread_v3.parallel_env()
    env.reset()
    
    #Environment and agent Settings
    agent_names = env.agents
    agent_mapping = {name: idx for idx, name in enumerate(agent_names)}
    obs_dim = env.observation_space(agent_names[0]).shape[0]
    act_dim = env.action_space(agent_names[0]).n
    num_agents = len(agent_names)
    
    #print(f" Environmental information: {num_agents} agents, observation dimension ={obs_dim}, action space size ={act_dim}")
    
    #Create the attention model and trainer
    attention_model = AttentionModule(obs_dim)
    attention_trainer = AttentionTrainer(attention_model)
    
    #Create an improved RIAL agent
    shared_agent = ImprovedRIALAgent(
        obs_dim, act_dim, num_agents,
        hidden_dim=64,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=50000,
        batch_size=64,
        tau=0.01
    )
    
    #Train RIAL
    rial_rewards, _, _ = train_improved_rial(
        env, shared_agent, agent_mapping, attention_model, attention_trainer, num_episodes=3000)
    
    #Draw and save the individual training curve of RIAL
    plt.figure(figsize=(10, 6))
    plt.plot(rial_rewards, label='Raw Reward', alpha=0.5)
    plt.plot(moving_average(rial_rewards), label='Moving Average (50)', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Improved RIAL Training Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rial_training_curve.png")  #Save individual curves
    plt.show()
    
    #Try to load the DQN results and compare them
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    try:
        import pickle
        # make the path relative to the script location
        dqn_rewards_path = script_dir / "dqn_rewards.pkl"
        output_plot_path = script_dir / "dqn_vs_rial_comparison.png"

        with open(dqn_rewards_path, 'rb') as f:
            dqn_rewards = pickle.load(f)
        
        # Generate and save the comparison plot
        plot = plot_training_results(dqn_rewards, rial_rewards)
        plot.savefig(output_plot_path)
        plot.show()

    except FileNotFoundError:
        print("Error: 'dqn_rewards.pkl' not found in the script's directory!")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    #Save the RIAL reward array
    try:
        output_path = script_dir / improved_rial_rewards.pkl

        # Save the rewards array
        with open(output_path, 'wb') as f:
            pickle.dump(rial_rewards, f)
        
        print(f"Successfully saved RIAL rewards to: {output_path}")
    
    except Exception as e:
        print(f"Failed to save RIAL rewards: {e}")
    
    print("everything will be ok!!!!!!!")
