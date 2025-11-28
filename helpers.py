import numpy as np
# import jax.numpy as jnp
import pickle as pkl
import matplotlib.pyplot as plt
from fa_env.envs.grid_world import GridWorldEnv
from scipy.spatial.distance import cdist
from datetime import datetime
from main import q_learning, q_learning_fa, sarsa


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

def fetch_model(is_training, env, path):
    if is_training or path == "":
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(path, "rb") as f:
            q = pkl.load(f)
    return q

def fetch_model_fa(is_training, env, path, featurizer):
    if is_training or path == "":
        W = np.random.randn(featurizer.n_features, env.action_space.n) * 0.01
    else:
        with open(path, "rb") as f:
            W = pkl.load(f)
    return W

def fetch_model_ac(is_training, env, path, featurizer):
    if is_training or path == "":
        W = np.random.randn(featurizer.n_features, env.action_space.n) * 0.01
    else:
        with open(path, "rb") as f:
            W = pkl.load(f)
    return W

def update_model(q,path):
    with open(path, "wb") as f:
            pkl.dump(q, f)

def plot_run(episodes, rewards_per_episode, path):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(sum_rewards)
    plt.title(f"Collection Robot {path.split("/")[2]} Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (last 100 avg)")
    plt.savefig(path)
    plt.close()

# TODO: Finish mine didnt work properly so not using it
def  softmaxProb(x, Theta):
    probs = 0
    return probs

# TODO: Finish mine didnt work properly so not using it
def softmaxPolicy(x, Theta):
    probs = softmaxProb(x, Theta)
    a = 0
    return a

# TODO: Finish mine didnt work properly so not using it
def logSoftmaxPolicyGradient(x, a, Theta):
    probs = softmaxProb(x, Theta)
    gradient = 0
    return gradient