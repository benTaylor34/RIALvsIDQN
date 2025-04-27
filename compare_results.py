import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from pettingzoo.mpe import simple_spread_v3
from statsmodels.stats.proportion import proportion_confint
import os 

from RIAL2 import ImprovedRIALAgent, AttentionModule, compute_attention_weighted_messages
from DQN import DQNAgent

class MARLExperiment:
    def __init__(self, num_episodes=1000, num_runs=5):
        self.num_episodes = num_episodes
        self.num_runs = num_runs
        self.env = simple_spread_v3.parallel_env()
        self.env.reset()
        
        # Environment parameters
        self.agent_names = list(self.env.agents)
        self.obs_dim = self.env.observation_space(self.agent_names[0]).shape[0]
        self.act_dim = self.env.action_space(self.agent_names[0]).n
        self.num_agents = len(self.agent_names)
        
        # Results storage
        self.results = {
            'rial': defaultdict(list),
            'dqn': defaultdict(list),
            'comparison': defaultdict(list)
        }
        
    def initialize_agents(self):
        """Initialize both types of agents"""
        # RIAL components
        self.attention_model = AttentionModule(self.obs_dim)
        self.rial_agent = ImprovedRIALAgent(
            self.obs_dim, self.act_dim, self.num_agents,
            hidden_dim=64, epsilon=0.05
        )
        
        # DQN agents
        self.dqn_agents = {
            name: DQNAgent(self.obs_dim, self.act_dim, epsilon=0.05)
            for name in self.agent_names
        }
        
    def run_episode(self, team_assignment, render=False):
        """Run one evaluation episode with mixed teams"""
        obs, _ = self.env.reset()
        self.rial_agent.reset_hidden()
        
        done = {a: False for a in self.env.agents}
        rewards = {'rial': 0, 'dqn': 0}
        metrics = {
            'collisions': 0,
            'distances': [],
            'attention_weights': []
        }
        
        while not all(done.values()):
            if render:
                self.env.render()
                
            # Get positions for distance calculation
            positions = []
            for agent in self.env.agents:
                if not done[agent]:
                    pos = obs[agent][:2] if len(obs[agent]) >= 2 else [0, 0]
                    positions.append(pos)
                else:
                    positions.append([np.nan, np.nan])
            
            positions = np.array(positions)
            
            # Calculate distances
            if len(positions) > 1:
                try:
                    diff = positions[:, np.newaxis] - positions
                    distances = np.sqrt(np.sum(diff**2, axis=-1))
                    np.fill_diagonal(distances, np.nan)
                    avg_distance = np.nanmean(distances)
                    metrics['distances'].append(avg_distance)
                    
                    collision_mask = (distances > 0) & (distances < 0.1)
                    metrics['collisions'] += np.sum(collision_mask) / 2
                except:
                    metrics['distances'].append(0)
            
            # RIAL communication
            try:
                messages, attn_weights = compute_attention_weighted_messages(
                    obs, self.env.agents, self.attention_model
                )
                if attn_weights:
                    metrics['attention_weights'].append(attn_weights)
            except:
                messages = {agent: np.zeros(self.obs_dim) for agent in self.env.agents}
            
            # Get actions
            actions = {}
            for agent in self.env.agents:
                if not done[agent]:
                    try:
                        if team_assignment[agent] == 'rial':
                            agent_idx = self.agent_names.index(agent)
                            actions[agent] = self.rial_agent.act(
                                obs[agent], agent_idx, messages[agent]
                            )
                        else:
                            actions[agent] = self.dqn_agents[agent].act(obs[agent])
                    except:
                        actions[agent] = 0
            
            # Step environment
            try:
                next_obs, step_rewards, terms, truncs, _ = self.env.step(actions)
                
                for agent in self.env.agents:
                    if not done[agent]:
                        rewards[team_assignment[agent]] += step_rewards[agent]
                
                obs = next_obs
                done = {a: terms[a] or truncs[a] for a in self.env.agents}
            except:
                break
            
        return rewards, metrics
    
    def run_comparative_experiment(self):
        """Main experimental loop"""
        for run in range(self.num_runs):
            print(f"Starting run {run+1}/{self.num_runs}")
            self.initialize_agents()
            
            for episode in tqdm(range(self.num_episodes)):
                try:
                    # Random team assignment
                    team_assignment = {}
                    agents = self.agent_names.copy()
                    np.random.shuffle(agents)
                    for i, agent in enumerate(agents):
                        team_assignment[agent] = 'rial' if i < len(agents)//2 else 'dqn'
                    
                    # Run episode
                    rewards, metrics = self.run_episode(
                        team_assignment, 
                        render=(episode % 100 == 0)
                    )
                    
                    # Store results
                    self.results['rial']['rewards'].append(rewards.get('rial', 0))
                    self.results['dqn']['rewards'].append(rewards.get('dqn', 0))
                    self.results['comparison']['collisions'].append(metrics.get('collisions', 0))
                    
                    if 'distances' in metrics and metrics['distances']:
                        self.results['comparison']['distances'].append(
                            np.nanmean(metrics['distances'])
                        )
                    else:
                        self.results['comparison']['distances'].append(0)
                    
                    # Determine winner
                    rial_reward = rewards.get('rial', 0)
                    dqn_reward = rewards.get('dqn', 0)
                    if rial_reward > dqn_reward + 0.1:
                        self.results['comparison']['wins'].append('rial')
                    elif dqn_reward > rial_reward + 0.1:
                        self.results['comparison']['wins'].append('dqn')
                    else:
                        self.results['comparison']['wins'].append('draw')
                    
                    # Store attention metrics
                    if 'attention_weights' in metrics and metrics['attention_weights']:
                        self._process_attention_metrics(metrics['attention_weights'])
                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    continue
    
    def _process_attention_metrics(self, attention_data):
        """Calculate attention statistics"""
        entropies = []
        for step_data in attention_data:
            for agent, weights in step_data.items():
                if weights:
                    probs = np.array(list(weights.values()))
                    probs /= probs.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
        
        if entropies:
            self.results['rial']['attention_entropy'].append(np.mean(entropies))
    
    def analyze_results(self):
        """Perform statistical analysis and visualization"""
        print("\n=== Statistical Analysis ===")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame({
            'rial_reward': self.results['rial']['rewards'],
            'dqn_reward': self.results['dqn']['rewards'],
            'collisions': self.results['comparison']['collisions'],
            'distance': self.results['comparison']['distances'],
            'winner': self.results['comparison']['wins']
        })
        
        # 1. Reward comparison
        rial_mean = np.mean(df['rial_reward'])
        dqn_mean = np.mean(df['dqn_reward'])
        t_stat, p_val = stats.ttest_rel(df['rial_reward'], df['dqn_reward'])
        cohens_d = (rial_mean - dqn_mean) / np.std(df['rial_reward'] - df['dqn_reward'], ddof=1)
        
        print(f"Reward Comparison:")
        print(f"  RIAL Mean: {rial_mean:.2f} ± {np.std(df['rial_reward']):.2f}")
        print(f"  DQN Mean: {dqn_mean:.2f} ± {np.std(df['dqn_reward']):.2f}")
        print(f"  t-test: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"  Cohen's d: {cohens_d:.2f}")
        
        # 2. Win rate analysis
        win_counts = df['winner'].value_counts()
        rial_win_ci = proportion_confint(
            win_counts.get('rial', 0), 
            len(df),
            alpha=0.05,
            method='wilson'
        )
        print(f"\nWin Rates:")
        print(f"  RIAL Wins: {win_counts.get('rial', 0)} ({rial_win_ci[0]*100:.1f}%-{rial_win_ci[1]*100:.1f}%)")
        print(f"  DQN Wins: {win_counts.get('dqn', 0)}")
        print(f"  Draws: {win_counts.get('draw', 0)}")
        
        # 3. Behavioral metrics
        print(f"\nBehavioral Metrics:")
        print(f"  Avg. Collisions/Episode: {np.mean(df['collisions']):.2f} ± {np.std(df['collisions']):.2f}")
        print(f"  Avg. Inter-Agent Distance: {np.mean(df['distance']):.2f} ± {np.std(df['distance']):.2f}")
        
        if 'attention_entropy' in self.results['rial']:
            attn_entropy = self.results['rial']['attention_entropy']
            print(f"  Attention Entropy: {np.mean(attn_entropy):.2f} ± {np.std(attn_entropy):.2f}")
        
        # Generate visualizations
        self._generate_plots(df)
    
    def _generate_plots(self, df):
        """Create and save all analysis visualizations separately"""
        # Create output directory if it doesn't exist
        os.makedirs("experiment_plots", exist_ok=True)

        # 1. Learning Curve Plot
        plt.figure(figsize=(10, 6))
        rolling_window = max(1, len(df)//20)
        df['rial_smooth'] = df['rial_reward'].rolling(rolling_window).mean()
        df['dqn_smooth'] = df['dqn_reward'].rolling(rolling_window).mean()
        plt.plot(df['rial_smooth'], label='RIAL', color='orange')
        plt.plot(df['dqn_smooth'], label='DQN', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title('Learning Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("experiment_plots", "learning_curves.png"))
        plt.close()

        # 2. Reward Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df[['rial_reward', 'dqn_reward']], 
                        palette=['orange', 'blue'])
        plt.title('Reward Distribution Comparison')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.savefig(os.path.join("experiment_plots", "reward_distribution.png"))
        plt.close()

        # 3. Win Rate Plot (Pie Chart)
        plt.figure(figsize=(8, 8))
        win_counts = df['winner'].value_counts()
        win_counts.plot.pie(
            autopct='%1.1f%%', 
            colors=['orange', 'blue', 'gray'],
            labels=['RIAL', 'DQN', 'Draw'],
            startangle=90
        )
        plt.title('Win/Draw Proportions')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join("experiment_plots", "win_rates.png"))
        plt.close()

        # 4. Behavioral Metrics Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[['collisions', 'distance']], 
                    palette=['red', 'green'])
        plt.title('Behavioral Metrics Comparison')
        plt.xticks([0, 1], ['Collisions per Episode', 'Average Inter-Agent Distance'])
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join("experiment_plots", "behavioral_metrics.png"))
        plt.close()

        # 5. Attention Analysis Plot (if available)
        if 'attention_entropy' in self.results['rial']:
            plt.figure(figsize=(10, 6))
            plt.hist(
                self.results['rial']['attention_entropy'], 
                bins=20, 
                color='purple', 
                alpha=0.7
            )
            plt.xlabel('Attention Weight Entropy')
            plt.ylabel('Frequency')
            plt.title('RIAL Attention Consistency')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join("experiment_plots", "attention_analysis.png"))
            plt.close()

        print("\nVisualizations saved to 'experiment_plots' directory:")

if __name__ == "__main__":
    experiment = MARLExperiment(
        num_episodes=500,
        num_runs=3
    )
    experiment.run_comparative_experiment()
    experiment.analyze_results()