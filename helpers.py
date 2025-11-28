import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import jax.numpy as jnp
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
        print(centers)
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

def fetch_ac_model(is_training, env, path, featurizer):
    if is_training or path == "":
        Theta = np.random.randn(featurizer.n_features, env.action_space.n) * 0.01
        w = np.random.randn(featurizer.n_features) * 0.01
        return Theta, w
    else:
        with open(path, "rb") as f:
            ac_dict = pkl.load(f)
    return ac_dict['actor'], ac_dict['critic']

def update_model(q,path):
    with open(path, "wb") as f:
            pkl.dump(q, f)

def update_ac_model(Theta, w ,path):
    with open(path, "wb") as f:
            pkl.dump({
                'actor'  : Theta,
                'critic' : w
            }, f)

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
def softmaxProb(x, Theta):
    h_s = jnp.dot(Theta.T, x)

    exponents = jnp.exp(h_s)
    sum_exp = jnp.sum(exponents)
    
    return exponents / sum_exp

# TODO: Finish mine didnt work properly so not using it
def softmaxPolicy(x, Theta):
    probabilities = softmaxProb(x, Theta)

    probabilities = np.array(probabilities, dtype=np.float64)
    prob_sum = np.sum(probabilities)
    probabilities = probabilities / prob_sum

    return np.random.choice(len(probabilities), p=probabilities)

# TODO: Finish mine didnt work properly so not using it
def logSoftmaxPolicyGradient(x, a, Theta):
    probs = softmaxProb(x, Theta)
    
    actions_n = Theta.shape[1]
    one_hot = np.zeros(actions_n)
    one_hot[a] = 1
    
    delta_vec = one_hot - probs
    gradient = jnp.outer(x, delta_vec)
    
    return gradient

def check_gradient(env):
    featurizer = RBFStateFeaturizer(env, 100)
    s = featurizer.featurize(env.observation_space.sample())
    a = env.action_space.sample()
    Theta = np.ones([featurizer.n_features, env.action_space.n])  # or any other initialization
    analytic_grads = logSoftmaxPolicyGradient(s, a, Theta)
    match_grad = logSoftmaxGradChecker(s, a, Theta, softmaxProb, analytic_grads)
    print(f'Gradient matched? {match_grad}')

def logSoftmaxGradChecker(s, a, Theta, softmaxProb, analytic_grads):
    def grad_test_func(Theta):
        return jnp.log(softmaxProb(s, Theta)[a])
    auto_grads = jax.grad(grad_test_func)(Theta)
    return np.allclose(analytic_grads, auto_grads)
