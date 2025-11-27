import numpy as np
import pickle as pkl
from fa_env.envs.grid_world import GridWorldEnv
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


#Helpers
def fetch_model(is_training, env, algorithm):
    if is_training or algorithm == "":
        q = np.zeros((env.observation_space.n, env.action_space.n))
        #q = np.random.randn(env.observation_space.n, env.action_space.n) * 0.01
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



class RbfFeaturizer():
    '''
        This class converts the raw state/obvervation features into
        RBF features. It does a z-score normalization and computes the
        Gaussian kernel values from randomly selected centers.
    '''

    def __init__(self, env, n_features=100):
        centers = np.array([env.observation_space.sample()
                            for _ in range(n_features)])
        self._mean = np.mean(centers, axis=0, keepdims=True)
        self._std = np.std(centers, axis=0, keepdims=True)
        self._centers = (centers - self._mean) / self._std
        self.n_features = n_features

    def featurize(self, state):
        z = state[None, :] - self._mean
        z = z / self._std
        dist = cdist(z, self._centers)
        return np.exp(- (dist) ** 2).flatten()

# Algorithms
def q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes): 
    q = fetch_model(is_training, env, "q_learning")
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
                #prevents overflow
                q[state, action] = np.clip(q[state, action], -1e6, 1e6)

            state = new_state

        #Update exploration rate, then lowering learning rate once fully greedy
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = total_reward

    #Post-training
    env.close()

    plot_run(episodes, rewards_per_episode, "q_learning")

    if is_training: 
        update_model(q,"q_learning")

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

    plot_run(episodes, rewards_per_episode, "q_learning_fa")
        



def sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes): 
    q = fetch_model(is_training, env, "sarsa")
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

        # # Update exploration rate, then lowering learning rate once fully greedy
        # epsilon = max(epsilon - epsilon_decay_rate, 0)
        # if epsilon == 0:
        #    learning_rate_a = 0.0001  # Lower learning rate once fully greedy

        rewards_per_episode[i] = rewards

    #Post-training
    env.close()

    plot_run(episodes, rewards_per_episode, "sarsa")

    if is_training: 
        update_model(q,"sarsa")


def run(episodes, is_training=True, render=False):
    env = GridWorldEnv(render_mode="human" if render else None)
    np.random.seed(101194261)

    #Algorithm control variables
    learning_rate_a = 0.1
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    max_model_steps = 10

    #Algorithm 1
    # q_learning(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes)
    #Algorithm 2
    #scipy version might cause issues
    # featurizer = RbfFeaturizer(env, 100)
    # q_learning_fa(is_training, env, featurizer, discount_factor, learning_rate_a, epsilon, epsilon_decay_rate, episodes, evaluate_every=20)
    
    #Algorithm 3
    sarsa(is_training, env, learning_rate_a, discount_factor, epsilon, epsilon_decay_rate, episodes)


#Main
# run(100000, render=False, is_training=True)
run(3, render=True, is_training=False)
