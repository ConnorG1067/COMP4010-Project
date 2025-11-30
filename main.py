#////////////////////////////////////////////
# Program: Comp 4010 Trash Collection Agent
# Author: Connor Gomes, Jackson Scott, Shawnia Noel, Antony Ren, Sean Xie
#///////////////////////////////////////////

#Imports
import numpy as np
import jax.numpy as jnp
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
                action = np.random.randint(0, env.action_space.n - 1)
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if(is_training):
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state, action] += learning_rate_a * (reward + discount_factor * np.max(q[new_state]) - q[state, action])

            state = new_state

        #Update exploration rate, then lowering learning rate once fully greedy
        epsilon = max(epsilon - epsilon_decay_rate, 0) # fast linear decline
        #epsilon = max(epsilon * epsilon_decay_rate, 0.01)  # slow exponential decline
        if epsilon == 0:
            learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = total_reward

    if is_training: 
        helpers.update_model(q, model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)

def q_learning_fa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path="", fig_path=""):
    featurizer = helpers.RbfFeaturizer(env, 100)
    W = helpers.fetch_model_fa(is_training, env, model_path, featurizer)
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(1, episodes + 1):
        s = env.reset()
        s = helpers.featurizer.featurize(s) # convert to a feature vector
        terminated = truncated = False
        total_reward = 0
        while not (terminated or truncated):
            if np.random.rand() < epsilon:
                a = np.random.randint(env.action_space.n)
            else:
                a = a = np.argmax(W.T @ s)

            next_state, reward, terminated, truncated, _ = env.step(a)
            feature_v = helpers.featurizer.featurize(next_state)
            total_reward += reward

            if(is_training):
                #TD_error = R + γ(Max(qhat(S'.a'))) - qhat(S.A)
                td_error = reward + discount_factor * np.max(np.dot(W.T, feature_v)) - np.dot(W[:, a].T, s)
                #Wa+1 = Wa + α(TD_error) * S
                W[:, a] = W[:, a] + learning_rate_a * td_error * s

            s = feature_v
        
        # Update exploration rate, then lowering learning rate once fully greedy
        epsilon = max(epsilon - epsilon_decay_rate, 0) # fast linear decline
        if epsilon == 0:
            learning_rate_a = 0.0001  # Lower learning rate once fully greedy
        
        rewards_per_episode[i] = total_reward

    if is_training: 
        helpers.update_model(W, model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)

def sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path="", fig_path=""): 
    q = helpers.fetch_model(is_training, env, model_path)
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
        if(is_training):
            epsilon = max(epsilon * epsilon_decay_rate, 0.01)
            if epsilon == 0:
                learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = rewards

    if is_training: 
        helpers.update_model(q,model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)


def actor_critic(is_training, env, discount_factor, episodes, actor_step_size=0.005, critic_step_size=0.005, model_path="", fig_path=""):
    featurizer = helpers.RbfFeaturizer()(env, 100)
    Theta, w = helpers.fetch_ac_model(is_training, env, model_path, featurizer)    # Initialize parameters / weights arbitrarily 
    
    rewards_per_episode = np.zeros(episodes)
    for i in range(1, episodes + 1):    # Loop forever (for each episode)
        s, info = env.ac_reset()    # Initialize S0
        s = featurizer.featurize(s)
        terminated = truncated = False  
        actor_discount = 1
        
        rewardTotal = 0
        while not (terminated or truncated):    # Loop for each step t = 0, 1, 2, ...., T of episode
            a = helpers.softmaxPolicy(s, Theta)    # Choose action A_t ~ π(⋅ | S_t;θ)
            newState, reward, terminated, truncated, moreInfo = env.step(a)   # Take action A_t, observe R_t+1, S_t+1
            newState = featurizer.featurize(newState)

            rewardTotal += reward
            if(is_training):

                sTranspose = np.transpose(s)
                currentValue = sTranspose @ w

                if (terminated == False):
                    newStateTranspose = np.transpose(newState)
                    newValue = newStateTranspose @ w
                else:
                    newValue = 0

                tdError = reward + (gamma * newValue) - currentValue    # Calculate squiggly thing (tdError)    R + γ(Max(qhat(S'.a'))) - qhat(S.A)    (ST;w) ≐ 0
                w += (critic_step_size * tdError * s) # Semi-grad update critic    w ← w + αw ⋅ δt ⋅ ∇vhat(St ; w)
                gradient = helpers.logSoftmaxPolicyGradient(s, a, Theta)

                Theta += (actor_step_size * tdError * actor_discount * gradient)    # Policy grad update actor    θ ← θ + αθ ⋅ δt ⋅ γt ⋅ ∇log π(At|St;θ)

                actor_discount *= discount_factor # should this remain in is training?

            s = newState

        rewards_per_episode[i] = rewards
        
    if is_training: 
        helpers.update_ac_model(Theta, w, model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)

def dyna_q(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, max_model_step, model_path="", fig_path=""): 
    q = helpers.fetch_model(is_training, env, model_path)
    rewards_per_episode = np.zeros(episodes)
    m = {}

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
                action = np.random.randint(0, env.action_space.n -1)
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if(is_training):
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state, action] += learning_rate_a * (reward + discount_factor * np.max(q[new_state]) - q[state, action])
                
                # Model(S,A) <- R, S'    
                m[(state, action)] = (reward, new_state)
                
                # PlanningUpdate(Q, Model)
                keys = list(m.keys())
                for i in range(max_model_step):
                    s, a = keys[np.random.choice(len(keys))] 
                    
                    # R, S' <-Model(S,A)
                    r, new_s = m[(s, a)]
                    q[s, a] += learning_rate_a * (r + discount_factor * np.max(q[new_s]) - q[s, a])


            state = new_state

        #Update exploration rate, then lowering learning rate once fully greedy
        epsilon = max(epsilon - epsilon_decay_rate, 0) # fast linear decline
        #epsilon = max(epsilon * epsilon_decay_rate, 0.01)  # slow exponential decline
        if epsilon == 0:
            learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = total_reward

    if is_training: 
        helpers.update_model(q, model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)


# Every-visit
def monte_carlo(is_training, env, discount_factor, epsilon, episodes, model_path="", fig_path=""):
    q = helpers.fetch_model(is_training, env, model_path)
    returns_sum = {}
    returns_count = {}
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()
        episode = []
        terminated = False
        truncated = False
        total_reward = 0
        if(i%1000==0):
            print(i)

        while not terminated and not truncated:
            if is_training and np.random.rand() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = new_state

        if is_training:
            G = 0
            for t in reversed(range(len(episode))):
                s, a, r = episode[t]
                G = discount_factor * G + r

                if (s, a) not in returns_sum:
                    returns_sum[(s, a)] = 0.0
                    returns_count[(s, a)] = 0
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1
                if s not in q:
                    q[s] = np.zeros(env.action_space.n)
                q[s, a] = returns_sum[(s, a)] / returns_count[(s, a)]


        rewards_per_episode[i] = total_reward

    if is_training:
        helpers.update_model(q, model_path)
        helpers.plot_run(episodes, rewards_per_episode, fig_path)




def run(episodes, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, max_model_step, is_training=True, render=False, model_path="", fig_path=""):
    env = GridWorldEnv(render_mode="human" if render else None)

    # Algorithms
    if("q_learning" in model_path):
        q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path, fig_path)
    elif("q_learning_fa" in model_path):
        q_learning_fa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path, fig_path)
    elif("sarsa" in model_path):
        sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, model_path, fig_path)
    elif("actor_critic" in model_path):
        # Doesnt work
        actor_critic(is_training, env, discount_factor, episodes, learning_rate_a, model_path=model_path, fig_path=fig_path) #learning_rate_a
    elif("dqn" in model_path):
        # Not finished
        pass
    elif("dyna_q" in model_path):
        dyna_q(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes, max_model_step, model_path, fig_path)
    elif("monti_carlo" in model_path):
        monte_carlo(is_training, env, discount_factor, epsilon, episodes, model_path, fig_path)
    
    # TODO : AFTER EVERYTHING EXPERIMENTS AND EASE OF PROGRAM SHIT

    env.close()


if __name__ == "__main__":
    iterations = 5000
    testing = True
 
    if(testing):
        # algorithms = ["q_learning", "sarsa", "q_learning_fa","dyna_q", "monti_carlo"]
        # algorithms = ["actor_critic"]
        algorithms = ["dyna_q"]
        #step_size_list = [0.0005, 0.001, 0.005, 0.01, 0.1]
        step_size_list = [0.1]
        discount_factor = 0.9
        epsilon_list = [0.9, 0.95, 1]
        epsilon_decay_rate = 0.0001
        max_model_step_list = [5, 10, 50]

        for algorithm in algorithms:
            for step_size in step_size_list:
                for epsilon in epsilon_list:
                    for steps in max_model_step_list:
                        run(
                            iterations, 
                            learning_rate_a = step_size,
                            discount_factor = discount_factor,
                            epsilon = epsilon,
                            epsilon_decay_rate = epsilon_decay_rate,  
                            max_model_step = steps,
                            render = False, 
                            is_training = True, 
                            model_path = f"./models/{algorithm}/{algorithm}_iter_{iterations}_lr_{step_size}_df_{discount_factor}_e_{epsilon}_edr_{epsilon_decay_rate}_ms_{steps}.pkl",
                            fig_path = f"./plots/{algorithm}/{algorithm}_iter_{iterations}_lr_{step_size}_df_{discount_factor}_e_{epsilon}_edr_{epsilon_decay_rate}_ms_{steps}.png",
                        )
                        if algorithm != "dyna_q":
                            break
                if algorithm == "monti_carlo":
                    break
    else:
        # When running with render mode pick the model path to run with
        running_model_path = "./models/monti_carlo/monti_carlo_iter_15000_lr_0.0005_df_0.9_e_1_edr_0.0001_ms_1.pkl"
        
        learning_rate_a = 0.1
        discount_factor = 0.9
        epsilon = 1.0
        epsilon_decay_rate = 0.0001
        max_model_step = 1

        run(
            3,
            learning_rate_a=learning_rate_a,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay_rate=epsilon_decay_rate, 
            max_model_step = max_model_step, 
            render=True, 
            is_training=False, 
            model_path=running_model_path,
        )


