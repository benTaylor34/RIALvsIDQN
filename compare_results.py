import numpy as np
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
    def __init__(self, num_episodes=500, num_runs=3):
        self.num_episodes = num_episodes
        self.num_runs = num_runs
        self.env = simple_spread_v3.parallel_env(N=4, local_ratio=0.5)
        self.env.reset()
        
        # environmnet parameters
        self.agent_names = list(self.env.agents)
        self.obs_dim = self.env.observation_space(self.agent_names[0]).shape[0]
        self.act_dim = self.env.action_space(self.agent_names[0]).n
        self.num_agents = len(self.agent_names)

        # make sure we have enough agents
        if self.num_agents < 4:
            raise ValueError("Environment must have at least 4 agents for this comparison")
        
        # minimum 2 of model on each team
        self.min_rial = 2
        self.min_dqn = 2

        # store resutls
        self.run_results = []
        self.aggregated_metrics = defaultdict(list)

    def initialize_agents(self):
        # RIAL 
        self.attention_model = AttentionModule(self.obs_dim)
        self.rial_agent = ImprovedRIALAgent(
            self.obs_dim, self.act_dim, self.num_agents,
            hidden_dim=64, epsilon=0.05
        )
        # DQN 
        self.dqn_agents = {
            name: DQNAgent(self.obs_dim, self.act_dim, epsilon=0.05)
            for name in self.agent_names
        }

    def assign_teams(self):
        # team assignment for teh agents
        team_assignment = {}
        agents = self.agent_names.copy()
        np.random.shuffle(agents)
        team_assignment[agents[0]] = 'rial'
        team_assignment[agents[1]] = 'rial'
        team_assignment[agents[2]] = 'dqn'
        team_assignment[agents[3]] = 'dqn'
        for agent in agents[4:]: # randomly assign remaining agents if there are any
            team_assignment[agent] = np.random.choice(['rial', 'dqn'])
            
        return team_assignment

    def run_episode(self, team_assignment, render=False):
        # run the evaluation episode
        obs, _ = self.env.reset()
        self.rial_agent.reset_hidden()
        done = {a: False for a in self.env.agents}
        rewards = {'rial': 0, 'dqn': 0}
        metrics = {
            'collisions': {
                'rial_rial': 0,
                'dqn_dqn': 0,
                'rial_dqn': 0
            },
            'distances': {
                'rial_rial': [],
                'dqn_dqn': [],
                'rial_dqn': []
            },
            'attention_weights': []
        }
        
        while not all(done.values()):
            if render:
                self.env.render()
            # get positions and team assignments
            positions = {}
            teams = {}
            for agent in self.env.agents:
                if not done[agent]:
                    pos = obs[agent][:2] if len(obs[agent]) >= 2 else [0, 0]
                    positions[agent] = np.array(pos, dtype=np.float32)
                    teams[agent] = team_assignment[agent]
                else:
                    positions[agent] = np.array([np.nan, np.nan])
                    teams[agent] = None
            
            # calc pairwise distance and collisions for the agents
            agents = list(positions.keys())
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    agent1 = agents[i]
                    agent2 = agents[j]
                    if teams[agent1] is None or teams[agent2] is None:
                        continue   
                    try:
                        distance = np.linalg.norm(positions[agent1] - positions[agent2])
                    except:
                        distance = np.inf
                    
                    # group by team combo
                    team_combo = f"{teams[agent1]}_{teams[agent2]}"
                    if team_combo in ['rial_rial', 'dqn_dqn', 'rial_dqn']:
                        metrics['distances'][team_combo].append(distance)
                        if distance < 0.1:#collision distance is any agents < 0.1 units from each other
                            metrics['collisions'][team_combo] += 1
            
            # RIAL communication and attention
            try:
                messages, attn_weights = compute_attention_weighted_messages(
                    obs, self.env.agents, self.attention_model
                )
                if attn_weights:
                    metrics['attention_weights'].append(attn_weights)
            except Exception as e:
                messages = {agent: np.zeros(self.obs_dim) for agent in self.env.agents}
            
            # get actions to do
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
            
            # next step in environment
            try:
                next_obs, step_rewards, terms, truncs, _ = self.env.step(actions)
                
                for agent in self.env.agents:
                    if not done[agent]:
                        rewards[team_assignment[agent]] += step_rewards[agent]
                
                obs = next_obs
                done = {a: terms[a] or truncs[a] for a in self.env.agents}
            except Exception as e:
                print(f"Environment step error: {e}")
                break
        
        # calc mean distance
        for distance_type in metrics['distances']:
            if metrics['distances'][distance_type]:
                metrics['distances'][distance_type] = np.mean(metrics['distances'][distance_type])
            else:
                metrics['distances'][distance_type] = np.nan
        
        return rewards, metrics

    def run_comparative_experiment(self):
        """Main experimental loop"""
        for run in range(self.num_runs):
            print(f"\n=== Starting run {run+1}/{self.num_runs} ===")
            self.initialize_agents()
            run_data = {
                'rial_rewards': [],
                'dqn_rewards': [],
                'wins': [],
                'collisions': [],
                'distances': [],
                'attention_weights': []
            }
            
            for episode in tqdm(range(self.num_episodes), desc=f"Run {run+1}"):
                try:
                    team_assignment = self.assign_teams()#assign the teams
                    # run the episode
                    rewards, metrics = self.run_episode(team_assignment)
                    run_data['rial_rewards'].append(rewards['rial'])
                    run_data['dqn_rewards'].append(rewards['dqn'])
                    run_data['collisions'].append(metrics['collisions'])
                    run_data['distances'].append(metrics['distances'])
                    run_data['attention_weights'].append(metrics['attention_weights'])
                    
                    # get winner
                    if rewards['rial'] > rewards['dqn'] + 0.1:
                        run_data['wins'].append('rial')
                    elif rewards['dqn'] > rewards['rial'] + 0.1:
                        run_data['wins'].append('dqn')
                    else:
                        run_data['wins'].append('draw')
                        
                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    continue
            
            #store run data
            self.run_results.append(run_data)
            self._aggregate_run_metrics(run, run_data)

    def _aggregate_run_metrics(self, run_idx, run_data):
        self.aggregated_metrics['run'].append(run_idx)#save the run index
        
        # get rewaerds and metrics
        self.aggregated_metrics['rial_reward_mean'].append(np.mean(run_data['rial_rewards']))
        self.aggregated_metrics['rial_reward_std'].append(np.std(run_data['rial_rewards']))
        self.aggregated_metrics['dqn_reward_mean'].append(np.mean(run_data['dqn_rewards']))
        self.aggregated_metrics['dqn_reward_std'].append(np.std(run_data['dqn_rewards']))
        win_counts = pd.Series(run_data['wins']).value_counts()#winrate
        self.aggregated_metrics['rial_wins'].append(win_counts.get('rial', 0))
        self.aggregated_metrics['dqn_wins'].append(win_counts.get('dqn', 0))
        self.aggregated_metrics['draws'].append(win_counts.get('draw', 0))
        if run_data['collisions']:#collisions
            self.aggregated_metrics['rial_rial_collisions'].append(
                np.mean([c['rial_rial'] for c in run_data['collisions']])
            )
            self.aggregated_metrics['dqn_dqn_collisions'].append(
                np.mean([c['dqn_dqn'] for c in run_data['collisions']])
            )
            self.aggregated_metrics['cross_collisions'].append(
                np.mean([c['rial_dqn'] for c in run_data['collisions']])
            )
        if run_data['distances']:#distiances
            self.aggregated_metrics['rial_rial_distance'].append(
                np.nanmean([d['rial_rial'] for d in run_data['distances']])
            )
            self.aggregated_metrics['dqn_dqn_distance'].append(
                np.nanmean([d['dqn_dqn'] for d in run_data['distances']])
            )
            self.aggregated_metrics['cross_distance'].append(
                np.nanmean([d['rial_dqn'] for d in run_data['distances']])
            )

        if run_data['attention_weights']:#attention
            entropies = []
            for ep_weights in run_data['attention_weights']:
                for step_data in ep_weights:
                    for agent, weights in step_data.items():
                        if weights:
                            probs = np.array(list(weights.values()))
                            probs = probs / probs.sum()
                            entropy = -np.sum(probs * np.log(probs + 1e-10))
                            entropies.append(entropy)
            if entropies:
                self.aggregated_metrics['mean_attention_entropy'].append(np.mean(entropies))
            else:
                self.aggregated_metrics['mean_attention_entropy'].append(0)
        else:
            self.aggregated_metrics['mean_attention_entropy'].append(0)

    def analyze_results(self):
        if not self.run_results:
            print("No results to analyze")
            return
        
        #print the results of the differnt stats tests
        print("\n=== Statistical Analysis ===")
        agg_df = pd.DataFrame(self.aggregated_metrics)

        #rewards includig mean and std, min, max etc
        print("\n1. Reward Comparison Across Runs:")
        print(agg_df[['rial_reward_mean', 'dqn_reward_mean']].describe())
        
        t_stat, p_val = stats.ttest_rel(
            agg_df['rial_reward_mean'],
            agg_df['dqn_reward_mean']
        )
        cohens_d = (agg_df['rial_reward_mean'].mean() - agg_df['dqn_reward_mean'].mean()) / \
                   agg_df[['rial_reward_mean', 'dqn_reward_mean']].values.std(ddof=1)
        
        print(f"\nPaired t-test across runs:")
        print(f"  t = {t_stat:.2f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {cohens_d:.2f}")
        
        #win rate
        total_episodes = self.num_runs * self.num_episodes
        total_rial_wins = agg_df['rial_wins'].sum()
        total_dqn_wins = agg_df['dqn_wins'].sum()
        total_draws = agg_df['draws'].sum()
        
        print("\n2. Win Rates Across All Runs:")
        print(f"  RIAL Wins: {total_rial_wins} ({total_rial_wins/total_episodes*100:.1f}%)")
        print(f"  DQN Wins: {total_dqn_wins} ({total_dqn_wins/total_episodes*100:.1f}%)")
        print(f"  Draws: {total_draws} ({total_draws/total_episodes*100:.1f}%)")
        
        # collision and distance
        print("\n3. Behavioral Metrics Across Runs:")
        print("Collisions:")
        print(f"  RIAL-RIAL: {agg_df['rial_rial_collisions'].mean():.2f} ± {agg_df['rial_rial_collisions'].std():.2f}")
        print(f"  DQN-DQN: {agg_df['dqn_dqn_collisions'].mean():.2f} ± {agg_df['dqn_dqn_collisions'].std():.2f}")
        print(f"  Cross-Team: {agg_df['cross_collisions'].mean():.2f} ± {agg_df['cross_collisions'].std():.2f}")
        
        print("\nDistances:")
        print(f"  RIAL-RIAL: {agg_df['rial_rial_distance'].mean():.2f} ± {agg_df['rial_rial_distance'].std():.2f}")
        print(f"  DQN-DQN: {agg_df['dqn_dqn_distance'].mean():.2f} ± {agg_df['dqn_dqn_distance'].std():.2f}")
        print(f"  Cross-Team: {agg_df['cross_distance'].mean():.2f} ± {agg_df['cross_distance'].std():.2f}")
        
        # make visualizations
        self._generate_plots(agg_df)

    def _generate_plots(self, agg_df):
        os.makedirs("analysis_plots", exist_ok=True)
        
        #reward comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=agg_df[['rial_reward_mean', 'dqn_reward_mean']],
                    palette=['orange', 'blue'])
        plt.title('Reward Distribution Across Runs')
        plt.ylabel('Mean Reward per Run')
        plt.savefig("analysis_plots/reward_comparison.png")
        plt.close()
        
        # win rate pie chart
        plt.figure(figsize=(8, 6))
        wins = [
            agg_df['rial_wins'].sum(),
            agg_df['dqn_wins'].sum(),
            agg_df['draws'].sum()
        ]
        plt.pie(wins, labels=['RIAL', 'DQN', 'Draw'],
                autopct='%1.1f%%', colors=['orange', 'blue', 'gray'])
        plt.title('Win/Draw Proportions Across All Runs')
        plt.savefig("analysis_plots/win_rates.png")
        plt.close()
        
        #collision 
        plt.figure(figsize=(10, 6))
        collision_data = pd.DataFrame({
            'RIAL-RIAL': agg_df['rial_rial_collisions'],
            'DQN-DQN': agg_df['dqn_dqn_collisions'],
            'Cross-Team': agg_df['cross_collisions']
        })
        sns.boxplot(data=collision_data)
        plt.title('Collision Metrics Across Runs')
        plt.ylabel('Mean Collisions per Episode')
        plt.savefig("analysis_plots/collision_metrics.png")
        plt.close()
        
        # distance 
        plt.figure(figsize=(10, 6))
        distance_data = pd.DataFrame({
            'RIAL-RIAL': agg_df['rial_rial_distance'],
            'DQN-DQN': agg_df['dqn_dqn_distance'],
            'Cross-Team': agg_df['cross_distance']
        })
        sns.boxplot(data=distance_data)
        plt.title('Distance Metrics Across Runs')
        plt.ylabel('Mean Distance')
        plt.savefig("analysis_plots/distance_metrics.png")
        plt.close()
        
        # mean learning curves across all runs
        plt.figure(figsize=(12, 6))
        for run_data in self.run_results:
            plt.plot(pd.Series(run_data['rial_rewards']).rolling(20, min_periods=1).mean(),
                     color='orange', alpha=0.3)
            plt.plot(pd.Series(run_data['dqn_rewards']).rolling(20, min_periods=1).mean(),
                     color='blue', alpha=0.3)
        
        all_rial = np.array([run['rial_rewards'] for run in self.run_results])
        all_dqn = np.array([run['dqn_rewards'] for run in self.run_results])
        mean_rial = pd.DataFrame(all_rial.T).rolling(20, min_periods=1).mean().mean(axis=1)
        mean_dqn = pd.DataFrame(all_dqn.T).rolling(20, min_periods=1).mean().mean(axis=1)
        
        plt.plot(mean_rial, color='orange', linewidth=3, label='RIAL Mean')
        plt.plot(mean_dqn, color='blue', linewidth=3, label='DQN Mean')
        
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title('Learning Curves Across All Runs')
        plt.legend()
        plt.grid(True)
        plt.savefig("analysis_plots/learning_curves.png")
        plt.close()
        
        print("\nAnalysis plots saved to 'analysis_plots' directory")

if __name__ == "__main__":
    experiment = MARLExperiment(
        num_episodes=1500,
        num_runs=5
    )
    experiment.run_comparative_experiment()
    experiment.analyze_results()