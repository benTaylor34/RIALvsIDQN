import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dqn_agent import DQNAgent
from rial_agent import RIALAgent
import matplotlib.pyplot as plt
import random
import torch

# Set all random seeds
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # For CUDA
torch.backends.cudnn.benchmark = False

def print_reward_stats(rewards, agent_type):
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    max_episode = np.argmax(rewards)
    print(f"\n{agent_type} Results:")
    print(f"Average reward across all episodes: {avg_reward:.2f}")
    print(f"Highest reward: {max_reward:.2f} (Episode {max_episode})")
    return avg_reward, max_reward, max_episode

def train_dqn(env, episodes=1000, batch_size=32):
    agents = {
        agent: DQNAgent(
            env.observation_space(agent).shape[0],
            env.action_space(agent).n
        ) for agent in env.possible_agents
    }
    rewards_history = []

    for episode in range(episodes):
        observations, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        dones = {agent: False for agent in env.possible_agents}

        while not all(dones.values()):
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].act(observations[agent])

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            for agent in env.agents:
                agents[agent].remember(
                    observations[agent],
                    actions[agent],
                    rewards[agent],
                    next_observations[agent],
                    dones[agent]
                )
                episode_rewards[agent] += rewards[agent]

            observations = next_observations

            for agent in env.possible_agents:
                if len(agents[agent].memory) > batch_size:
                    agents[agent].replay(batch_size)

        avg_ep_reward = np.mean(list(episode_rewards.values()))
        rewards_history.append(avg_ep_reward)
        print(f"Episode {episode}, Avg Reward: {avg_ep_reward:.2f}", end='\r')

    return rewards_history

def train_rial(env, episodes=1000, batch_size=32):
    agents = {
        agent: RIALAgent(
            env.observation_space(agent).shape[0],
            env.action_space(agent).n
        ) for agent in env.possible_agents
    }
    rewards_history = []

    for episode in range(episodes):
        observations, _ = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        dones = {agent: False for agent in env.possible_agents}
        last_messages = {agent: np.array([0]) for agent in env.possible_agents}

        while not all(dones.values()):
            actions = {}
            messages = {}
            for agent in env.agents:
                actions[agent], messages[agent] = agents[agent].act(observations[agent], last_messages[agent])

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

            for agent in env.agents:
                agents[agent].remember(
                    observations[agent],
                    actions[agent],
                    messages[agent],
                    rewards[agent],
                    next_observations[agent],
                    dones[agent]
                )
                episode_rewards[agent] += rewards[agent]
                last_messages[agent] = np.array([messages[agent]])

            observations = next_observations

            for agent in env.possible_agents:
                if len(agents[agent].memory) > batch_size:
                    agents[agent].replay(batch_size)

        avg_ep_reward = np.mean(list(episode_rewards.values()))
        rewards_history.append(avg_ep_reward)
        print(f"Episode {episode}, Avg Reward: {avg_ep_reward:.2f}", end='\r')

    return rewards_history

if __name__ == "__main__":
    # Initialize the Parallel API environment
    env = simple_spread_v3.parallel_env(N=4, max_cycles=50) # 4 agents
    env.reset(seed=SEED)# reset the environment

    # Train RIAL
    print("\nTraining RIAL...")
    rial_rewards = train_rial(env, episodes=500)
    rial_avg, rial_max, rial_max_ep = print_reward_stats(rial_rewards, "RIAL")

    # Reset environment for DQN training to compare with RIAL
    env = simple_spread_v3.parallel_env(N=4, max_cycles=50)
    env.reset(seed=SEED)

    # Train DQN
    print("\nTraining DQN...")
    dqn_rewards = train_dqn(env, episodes=500)
    dqn_avg, dqn_max, dqn_max_ep = print_reward_stats(dqn_rewards, "DQN")

    
  
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_rewards, label=f"DQN (Avg: {dqn_avg:.2f}, Max: {dqn_max:.2f} @ Ep.{dqn_max_ep})")
    plt.plot(rial_rewards, label=f"RIAL (Avg: {rial_avg:.2f}, Max: {rial_max:.2f} @ Ep.{rial_max_ep})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("DQN vs RIAL Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("results.png")
    plt.show()