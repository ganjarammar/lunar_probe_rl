import os
import argparse
import numpy as np
from datetime import datetime
from lunar_env import LunarEnvironment
from model import LunarLanderAgent
import matplotlib.pyplot as plt

def train(args):
    env = LunarEnvironment()
    agent = LunarLanderAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Setup direktori untuk checkpoint
    checkpoint_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    rewards_history = []
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.get_action(state, noise_scale=0.1)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Training step
            # ... implementasi algoritma training ...
            
            if done:
                break
            
            state = next_state
            
            if args.render:
                env.render()
        
        rewards_history.append(episode_reward)
        
        # Simpan checkpoint
        if episode % args.save_interval == 0:
            agent.save(f"{checkpoint_dir}/ep_{episode}")
        
        print(f"Episode {episode}: Reward = {episode_reward}")
    
    # Plot hasil training
    plt.plot(rewards_history)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('metrics/training_progress.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    train(args) 