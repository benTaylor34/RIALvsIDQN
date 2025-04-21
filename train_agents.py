import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dqn_agent import DQNAgent
from rial_agent import RIALAgent
import matplotlib.pyplot as plt

def train_dqn(env, episodes=1000, batch_size=32):
    agents = [DQNAgent(env.observation_space(agent).shape[0], env.action_space(agent).n) for agent in env.agents]
    rewards_history = []
    
    for episode in range(episodes):
        env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}
        
        for agent in env.agent_iter():
            obs, reward, done, _ = env.last()
            action = agents[env.agents.index(agent)].act(obs)
            env.step(action)
            episode_rewards[agent] += reward
            
            if done:
                break
        
        # Train each agent
        for agent in env.agents:
            if len(agents[env.agents.index(agent)].memory) > batch_size:
                agents[env.agents.index(agent)].replay(batch_size)
        
        rewards_history.append(np.mean(list(episode_rewards.values())))
        print(f"Episode {episode}, Avg Reward: {rewards_history[-1]}")
    
    return rewards_history

def train_rial(env, episodes=1000, batch_size=32):
    agents = [RIALAgent(env.observation_space(agent).shape[0], env.action_space(agent).n) for agent in env.agents]
    rewards_history = []
    
    for episode in range(episodes):
        env.reset()
        episode_rewards = {agent: 0 for agent in env.agents}
        last_messages = {agent: 0 for agent in env.agents}
        
        for agent in env.agent_iter():
            obs, reward, done, _ = env.last()
            action, message = agents[env.agents.index(agent)].act(obs, last_messages[agent])
            env.step(action)
            episode_rewards[agent] += reward
            last_messages[agent] = message
            
            if done:
                break
        
        # Train each agent
        for agent in env.agents:
            if len(agents[env.agents.index(agent)].memory) > batch_size:
                agents[env.agents.index(agent)].replay(batch_size)
        
        rewards_history.append(np.mean(list(episode_rewards.values())))
        print(f"Episode {episode}, Avg Reward: {rewards_history[-1]}")
    
    return rewards_history

if __name__ == "__main__":
    env = simple_spread_v3.parallel_env(N=2, max_cycles=50)  # 2 agents
    
    # Train DQN
    print("Training DQN...")
    dqn_rewards = train_dqn(env, episodes=500)
    
    # Train RIAL
    print("Training RIAL...")
    rial_rewards = train_rial(env, episodes=500)
    
    # Plot results
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(rial_rewards, label="RIAL")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig("results.png")
    plt.show()