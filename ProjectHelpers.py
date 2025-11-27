import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from fa_env.envs.grid_world import GridWorldEnv
from scipy.spatial.distance import cdist
from datetime import datetime


###########################################################################################################
# Experiment Classes
###########################################################################################################

def q_learning_experiments(env, episodes=1000):
    parameters = [
        {"learning_rate_a": 0.1, "discount_factor": 0.9, "epsilon": 1.0, "epsilon_decay_rate" : 0.0001},
        {"learning_rate_a": 0.05, "discount_factor": 0.9, "epsilon": 1.0, "epsilon_decay_rate" : 0.0001},
        {"learning_rate_a": 0.01, "discount_factor": 0.9, "epsilon": 1.0, "epsilon_decay_rate" : 0.0001}
    ]

    plt.figure(figsize=(10, 6))

    for params in parameters:
        print(params)
        _, rewards = q_learning(
            is_training=True,
            env=env,
            learning_rate_a=params["learning_rate_a"],
            discount_factor=params["discount_factor"],
            epsilon=params["epsilon"],
            epsilon_decay_rate=params["epsilon_decay_rate"],
            episodes=episodes
        )

        # Smooth rewards using moving average
        window = 20
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

        label = f"LR={params['learning_rate_a']}, DF={params['discount_factor']}"
        plt.plot(smoothed_rewards, label=label)

    plt.title("Q-Learning Performance with Different Parameters")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"q_learning_returns_{timestamp}.png"
    plt.savefig(filename)












###########################################################################################################
# Helper Classes
###########################################################################################################
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


###########################################################################################################
# Helper Functions
###########################################################################################################
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