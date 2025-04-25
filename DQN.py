import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import pickle  


#Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64,
                 memory_size=10000, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        q_next = self.qnetwork_target(next_states).detach()
        q_target = rewards + self.gamma * torch.max(q_next, dim=1)[0] * (~dones)

        q_expected = self.qnetwork_local(states).gather(1, actions).squeeze()
        loss = self.loss_fn(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())


# Training Loop
def train_multi_agent_dqn(env, agents, num_episodes=3000):
    reward_log = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=None)
        done = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.agents}

        while not all(done.values()):
            actions = {agent: agents[agent].act(obs[agent]) for agent in env.agents if not done[agent]}
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
                if not done[agent]:
                    agents[agent].step(obs[agent], actions[agent], rewards[agent],
                                       next_obs[agent], terminations[agent] or truncations[agent])
                    episode_reward[agent] += rewards[agent]

            obs = next_obs
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

        if episode % 10 == 0:
            for agent in agents.values():
                agent.update_target_network()

        if episode % 100 == 0:
            total = sum(episode_reward.values())
            print(f"Episode {episode}, Total Reward: {total:.2f}, Epsilon: {list(agents.values())[0].epsilon:.4f}")

        reward_log.append(sum(episode_reward.values()))

    return reward_log


#Utility

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    env = simple_spread_v3.parallel_env()
    env.reset()

    agents = {}
    for agent_name in env.agents:
        obs_size = env.observation_space(agent_name).shape[0]
        act_size = env.action_space(agent_name).n
        agents[agent_name] = DQNAgent(obs_size, act_size)

    rewards = train_multi_agent_dqn(env, agents, num_episodes=3000)

    
    #Plot and Save Results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Raw Reward", alpha=0.5)
    plt.plot(moving_average(rewards, window_size=50), label="Moving Average (50)", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Multi-Agent DQN on simple_spread_v3")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('/Users/xuzhengyi/Desktop/python/Scc462/dqn_reward_plot.png')
    plt.show()

    with open('/Users/xuzhengyi/Desktop/python/Scc462/dqn_rewards.pkl', 'wb') as f:
        pickle.dump(rewards, f)

    print("everything will be ok!!!!!!")
