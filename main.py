import numpy as np
import pickle as pkl
from fa_env.envs.grid_world import GridWorldEnv
import matplotlib.pyplot as plt


#Helpers
def fetch_model(is_training, env, algorithm):
    if is_training or algorithm == "":
        q = np.zeros((env.size**2, env.action_space.n))
    else:
        with open(f"./{algorithm}_model.pkl", "rb") as f:
            q = pkl.load(f)
    return q

def update_model(q,algorithm):
    with open(f"{algorithm}_model.pkl", "wb") as f:
            pkl.dump(q, f)

def plot_run(episodes, rewards_per_episode, algorithm):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.title(f"Collection Robot \"{algorithm}\"Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (last 100 avg)")
    plt.savefig(f'{algorithm}_model.png')
    plt.close()

def convert_action_to_int(array):
    return (array[0] + 1) * (array[1] + 1) - 1


# Algorithms
def q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes): 
    
    q = fetch_model(is_training, env,"q_learning")
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset environment 
        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards = 0
        while not truncated and not terminated:
            # Epsilon greedy algorithm
            if(is_training and np.random.rand() < epsilon):
                # If random number less than epsilon sample the action space uniformly
                action = env.action_space.sample()
            else:
                action = np.argmax(q[convert_action_to_int(state['agent_pos']), :])
                
                # print(q[convert_action_to_int(state['agent']), :])
                # print(action)
                # print()


            new_state, reward, terminated, truncated, _ = env.step(action)
            # is info which is used in an action mask should we make one
            # used for only picking legal option


            rewards += reward

            if(is_training):
                state_as_int = convert_action_to_int(state['agent_pos'])
                new_state_as_int = convert_action_to_int(new_state['agent_pos'])
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state_as_int, action] = q[state_as_int, action] + learning_rate_a * (reward + (discount_factor * np.max(q[new_state_as_int, :])) - q[state_as_int, action])
                
            
            state = new_state

        #Update exploration rate, then lowering learning rate once fully greedy
        #epsilon = max(epsilon - epsilon_decay_rate, 0)
        #if epsilon == 0:
        #    learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = rewards

    #Post-training
    env.close()

    plot_run(episodes, rewards_per_episode, "q_learning")

    if is_training: 
        update_model(q,"q_learning")

    
        

#def dyna_q(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, max_model_steps, episodes):

def sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes): 
    
    q = fetch_model(is_training, env, "sarsa")
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset environment 
        state = env.reset()[0]
        terminated = False
        truncated = False

        rewards = 0

        # Epsilon greedy algorithm
        if(is_training and np.random.rand() < epsilon):
            # If random number less than epsilon sample the action space uniformly
            action = env.action_space.sample()
        else:
            action = np.argmax(q[convert_action_to_int(state['agent_pos']), :])

        while not truncated and not terminated:
            new_state, reward, terminated, truncated, _ = env.step(action)
            # is info which is used in an action mask should we make one
            # used for only picking legal option

            # Do epsilon greedy to update action
            if(is_training and np.random.rand() < epsilon):
                # If random number less than epsilon sample the action space uniformly
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q[convert_action_to_int(state['agent_pos']), :])

            rewards += reward

            if(is_training):
                state_as_int = convert_action_to_int(state['agent_pos'])
                new_state_as_int = convert_action_to_int(new_state['agent_pos'])
                # Q(s, a) ← Q(s, a) + α * [r + γ * max(Q(s', ·)) - Q(s, a)]
                q[state_as_int, action] = q[state_as_int, action] + learning_rate_a * (reward + (discount_factor * np.max(q[new_state_as_int, new_action])) - q[state_as_int, action])
                
            state = new_state
            action = new_action

        #Update exploration rate, then lowering learning rate once fully greedy
        #epsilon = max(epsilon - epsilon_decay_rate, 0)
        #if epsilon == 0:
        #    learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = rewards

    #Post-training
    env.close()

    plot_run(episodes, rewards_per_episode, "sarsa")

    if is_training: 
        update_model(q,"sarsa")

def ActorCritic(env, discountFactor, actorStepSize, criticStepSize, maxEpisodes, evaluateEvery)


def run(episodes, is_training=True, render=False):
    env = GridWorldEnv(render_mode="human" if render else None)
    np.random.seed(101194261)

    #Algorithm control variables
    learning_rate_a = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    max_model_steps = 10

    #Algorithm 1
    q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes)
    #Algorithm 2
    #dyna_q(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, max_model_steps, episodes)
    #Algorithm 3
    # sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes)


#Main
run(100, render=False, is_training=True)
run(5, render=True)
