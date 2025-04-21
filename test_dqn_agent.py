import numpy as np
import torch
from dqn_agent import DQNAgent

def test_network_architecture():
    print("Testing Network Architecture...")
    state_dim = 4  # Example: CartPole state
    action_dim = 2  # Example: Left/Right
    agent = DQNAgent(state_dim, action_dim)
    
    # Check model input/output dimensions
    dummy_input = torch.randn(1, state_dim)
    output = agent.model(dummy_input)
    assert output.shape == (1, action_dim), f"Expected shape (1, {action_dim}), got {output.shape}"
    print("✅ Network Architecture Test Passed!")

def test_replay_buffer():
    print("Testing Replay Buffer...")
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim)
    
    # Add dummy experiences
    for _ in range(10):
        state = np.random.rand(state_dim)
        action = np.random.randint(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = np.random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)
    
    assert len(agent.memory) == 10, f"Expected 10 experiences, got {len(agent.memory)}"
    print("✅ Replay Buffer Test Passed!")

def test_training_loop():
    print("Testing Training Loop...")
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim)
    
    # Fill replay buffer
    for _ in range(100):
        state = np.random.rand(state_dim)
        action = np.random.randint(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = np.random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)
    
    # Train and check loss
    loss = agent.replay(batch_size=32)
    assert loss is not None, "Training did not occur (not enough samples?)"
    assert not torch.isnan(torch.tensor(loss)), "Training resulted in NaN loss!"
    print("✅ Training Loop Test Passed!")

def test_epsilon_decay():
    print("Testing Epsilon Decay...")
    state_dim = 4
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim, epsilon=1.0, epsilon_decay=0.99)
    
    initial_epsilon = agent.epsilon
    # Need to add experiences to memory first
    for _ in range(32):
        state = np.random.rand(state_dim)
        action = np.random.randint(action_dim)
        reward = np.random.rand()
        next_state = np.random.rand(state_dim)
        done = np.random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)
    
    for _ in range(10):
        agent.replay(batch_size=32)  # Decay happens in replay
    
    assert agent.epsilon < initial_epsilon, "Epsilon did not decay!"
    print("✅ Epsilon Decay Test Passed!")

if __name__ == "__main__":
    test_network_architecture()
    test_replay_buffer()
    test_training_loop()
    test_epsilon_decay()
    print("All tests passed! DQN Agent is working correctly.")