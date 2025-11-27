#////////////////////////////////////////////
# Program: Comp 4010 Trash Collection Agent
# Author: Connor Gomes, Jackson Scott, Shawnia Noel, Antony Ren, Sean Xie
#///////////////////////////////////////////

#Imports
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from fa_env.envs.grid_world import GridWorldEnv
from datetime import datetime
import helpers


# Algorithms
def q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path="", fig_path=""): 
    q = helpers.fetch_model(is_training, env, model_path)
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset environment 
        state = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        if(i%1000==0):
            print(i)

        while not truncated and not terminated:
            # Epsilon greedy algorithm
            if(is_training and np.random.rand() < epsilon):
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if(is_training):
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state, action] += learning_rate_a * (reward + discount_factor * np.max(q[new_state]) - q[state, action])
                #prevents overflow
                q[state, action] = np.clip(q[state, action], -1e6, 1e6)

            state = new_state

        # Update exploration rate, then lowering learning rate once fully greedy
        # if(is_training):
            # epsilon = max(epsilon * epsilon_decay_rate, 0.01)
            # if epsilon == 0:
            #     learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = total_reward

    if is_training: 
        helpers.update_model(q,f"./models/q-learning/q_learning_iteration_{episodes}.pkl")
    
    helpers.plot_run(episodes, rewards_per_episode, fig_path)

    return q, rewards_per_episode


def q_learning_fa(is_training, env, featurizer, discount_factor, learning_rate_a, epsilon, epsilon_decay_rate, episodes):
    W = np.random.randn(featurizer.n_features, env.action_space.n) * 0.01
    #q = fetch_model(is_training, env, "q_learningfa")
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(1, episodes + 1):
        s = env.reset()[0]
        s = featurizer.featurize(s) # convert to a feature vector
        terminated = truncated = False
        total_reward = 0
        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                a = np.random.randint(env.action_space.n)
            else:
                a = a = np.argmax(W.T @ s)

            next_state, reward, terminated, truncated, _ = env.step(a)
            feature_v = featurizer.featurize(next_state)
            total_reward += reward

            if(is_training):
                #TD_error = R + γ(Max(qhat(S'.a'))) - qhat(S.A)
                td_error = reward + discount_factor * np.max(np.dot(W.T, feature_v)) - np.dot(W[:, a].T, s)
                #Wa+1 = Wa + α(TD_error) * S
                W[:, a] = W[:, a] + learning_rate_a * td_error * s

            s = feature_v
        
        # Update exploration rate, then lowering learning rate once fully greedy
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001  # Lower learning rate once fully greedy
        
        rewards_per_episode[i] = total_reward

    helpers.plot_run(episodes, rewards_per_episode, "q_learning_fa")
        



def sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes): 
    q = helpers.fetch_model(is_training, env, "sarsa")
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset environment 
        state = env.reset()
        terminated = False
        truncated = False

        rewards = 0

        if(i%1000==0):
            print(i)

        # Choose initial action using epsilon-greedy
        if is_training and np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])

        while not truncated and not terminated:
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Do epsilon greedy to update action
            if(is_training and np.random.rand() < epsilon):
                # If random number less than epsilon sample the action space uniformly
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q[new_state, :])

            rewards += reward

            if(is_training):
                # Q(s,a) ← Q(s,a) + α[r + γ * Q(s′,a′) − Q(s,a)]
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor * q[new_state, new_action] - q[state, action])

                
            state = new_state
            action = new_action

        # Update exploration rate, then lowering learning rate once fully greedy
        # if(is_training):
            # epsilon = max(epsilon * epsilon_decay_rate, 0.01)
            # if epsilon == 0:
            #     learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = rewards

    helpers.plot_run(episodes, rewards_per_episode, "sarsa")

    if is_training: 
        helpers.update_model(q,"sarsa")









def run(episodes, is_training=True, render=False, model_path="", fig_path=""):
    env = GridWorldEnv(render_mode="human" if render else None)
    # np.random.seed(101194261)

    # Algorithm control variables
    learning_rate_a = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    max_model_steps = 10

    # Algorithms
    q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path, fig_path)
    # sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes)
    
    
    # Control Variables Experiments
    # helpers.q_learning_experiments(env, episodes)

    env.close()


if __name__ == "__main__":
    iterations = 10000
    testing = True

    if(testing):
        run(
            iterations, 
            render=False, 
            is_training=True, 
            model_path=f"./models/q-learning/q_learning_iteration_{iterations}.pkl",
            fig_path=f"./plots/q-learning/q_learning_iteration_{iterations}.png"
        )
    else:
        run(
            3, 
            render=True, 
            is_training=False, 
            model_path=f"./models/q-learning/q_learning_iteration_{iterations}.pkl",
        )


