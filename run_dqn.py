from fa_env.envs.grid_world import GridWorldEnv
from dqn import DoubleDQNAgent
import helpers
import numpy as np
import matplotlib.pyplot as plt



def ddqn_training_loop(env, agent, episodes):
  every_episode_rewards = []
  
  
  for ep in range(episodes):
    state_index, obs = env.reset()
    state = obs['dqn_obs']
    done = False
    episode_reward = 0
    
    while not done:
      action = agent.select_action(state)
      next_state_index, reward, terminated, truncated, obs = env.step(action)
      next_state = obs['dqn_obs']
      done = terminated or truncated

      transition = (state, action, reward, next_state, done)
      agent.store_experience(transition)
      agent.train()

      state = next_state
      episode_reward += reward

    
    
    every_episode_rewards.append(episode_reward)
    
  agent.save_model()
  print("\nTraining complete! Model saved.\n")
  
  return every_episode_rewards

def run_trained_ddqn(env, agent, episodes = 2):
  for ep in range(episodes):
    state_index, obs = env.reset()
    state = obs['dqn_obs']
    done = False
    episode_reward = 0

    while not done:
      action = agent.select_action(state)
      next_state_index, reward, terminated, truncated, obs = env.step(action)
      next_state = obs['dqn_obs']
      done = terminated or truncated

      state = next_state
      episode_reward += reward

    
    


def run(episodes, is_training=True, render=False):
  env = GridWorldEnv(render_mode="human" if render else None)
  state_index, obs = env.reset()
  input_shape = obs['dqn_obs'].shape[0]
  num_actions = env.action_space.n
  
  agent = DoubleDQNAgent(input_shape, num_actions)
  if is_training:
    rewards = ddqn_training_loop(env, agent, episodes)

    env.close()
    return rewards
  else:
    agent.load_model()
    agent.epsilon = 0.0
    run_trained_ddqn(env, agent, episodes)
    env.close()

run(episodes=1000, is_training=True, render=False)

