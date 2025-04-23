import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dqn_agent import DQNAgent
from rial_agent import RIALAgent
import matplotlib.pyplot as plt


def train_dqn(env, episodes=1000, batch_size=32):
    rewards_history = []
    for episode in range(episodes):
        env.reset()
        agents = [DQNAgent(env.observation_space(agent).shape[0], env.action_space(agent).n) for agent in env.agents]
        episode_rewards = {agent: 0 for agent in env.agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                env.step(None)
                continue

            action = agents[env.agents.index(agent)].act(obs)
            env.step(action)
            episode_rewards[agent] += reward

        rewards_history.append(np.mean(list(episode_rewards.values())))
        print(f"[DQN] Episode {episode}, Avg Reward: {rewards_history[-1]:.2f}")
    return rewards_history


def train_rial(env, episodes=1000, batch_size=32):
    rewards_history = []
    for episode in range(episodes):
        env.reset()
        agents = [RIALAgent(env.observation_space(agent).shape[0], env.action_space(agent).n) for agent in env.agents]
        episode_rewards = {agent: 0 for agent in env.agents}
        last_messages = {agent: 0 for agent in env.agents}

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                env.step(None)
                continue

            action, message = agents[env.agents.index(agent)].act(obs, last_messages[agent])
            env.step(action)
            episode_rewards[agent] += reward
            last_messages[agent] = message

        rewards_history.append(np.mean(list(episode_rewards.values())))
        print(f"[RIAL] Episode {episode}, Avg Reward: {rewards_history[-1]:.2f}")
    return rewards_history


if __name__ == "__main__":
    env = simple_spread_v3.env(N=2, max_cycles=50)

    print("Training DQN...")
    dqn_rewards = train_dqn(env, episodes=300)

    print("Training RIAL...")
    rial_rewards = train_rial(env, episodes=300)

    # Plot results
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(rial_rewards, label="RIAL")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("DQN vs RIAL Reward Comparison")
    plt.legend()
    plt.savefig("results.png")
    plt.show()
