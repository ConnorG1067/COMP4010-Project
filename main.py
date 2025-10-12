import numpy as np
import gymnasium as gym
import pickle as pkl
from fa_env.envs.grid_world import GridWorldEnv
import matplotlib.pyplot as plt
import time

def convert_action_to_int(array):
    return (array[0] + 1) * (array[1] + 1) - 1

def run(episodes, is_training=True, render=False):
    env = GridWorldEnv(render_mode="human" if render else None)

    if is_training:
        q = np.zeros((env.size**2, env.action_space.n))
    else:
        with open('./model.pkl', 'rb') as f:
            q = pkl.load(f)

    learning_rate_a = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards = 0
        # print(q)
        while not truncated and not terminated:
            # Epsilon greedy algorithm
            if(is_training and rng.random() < epsilon):
                # If random number less than epsilon sample the action space uniformly
                action = env.action_space.sample()
            else:
                print(state['agent'])
                action = np.argmax(q[convert_action_to_int(state['agent']), :])
                # print(q[convert_action_to_int(state['agent']), :])
                # print(action)
                # print()


            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            if(is_training):
                state_as_int = convert_action_to_int(state['agent'])
                new_state_as_int = convert_action_to_int(new_state['agent'])
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state_as_int, action] = q[state_as_int, action] + learning_rate_a * (reward + (discount_factor * np.argmax(q[new_state_as_int, :])) - q[state_as_int, action])
            
            state = new_state

        # Epsilon decay rate (As you develop a good policy you should explore less and take the greedy option)
        # If epsilon is ever 0 decrease learning rate

        rewards_per_episode[i] = rewards
    
    env.close()
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    # Plot and save the graph
    plt.plot(sum_rewards)
    plt.title("GridDot Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (last 100 avg)")
    plt.savefig('model.png')
    plt.close()

    if(is_training):
        with open("model.pkl", "wb") as f:
            pkl.dump(q, f)

# run(15000, render=False, is_training=True)

run(100, render=True, is_training=True)
