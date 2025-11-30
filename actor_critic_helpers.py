import numpy as np
from fa_env.envs.grid_world import GridWorldEnv
import matplotlib.pyplot as plt

# Helper functions for actor-critic algorithm start here
def ac_initialize(states, actions):    # Initialize parameters / weights arbitrarily 
    Theta = np.random.rand(states, actions) 
    w = np.random.rand(actions)
    return Theta, w

def initialize_s(env):    # Initialize S0
    s, info = env.reset()    
    return s

def softmax_prob(state, Theta):    # Calculate probabilities based on softmax
    transposeTheta = np.transpose(Theta)    
    h = transposeTheta @ state
    m = np.amax(h)
    robustH = h - m
    probs = np.exp(robustH) / np.sum(np.exp(robustH))  
    return probs

def softmax_policy(state, Theta):    # Acquire action sampled from softmax probabilities
    probs = softmaxProb(state, Theta)
    probsAmount = len(probs)
    a = np.random.choice(probsAmount, p=probs)  
    return a

def log_softmax_policy_gradient(state, a, Theta):    # Calculate softmax policy gradient
    probs = softmaxProb(state, Theta)
    actions = Theta.shape[1]   
    temp = np.zeros(actions)
    temp[a] = 1
    negativeProbs = temp - probs
    gradient = np.outer(state, negativeProbs)  
    return gradient

def take_action_observe(env, a, w):    # Take action At, observe Rt+1, St+1
    newState, reward, terminated, truncated, moreInfo = env.step(a)   
    sTranspose = np.transpose(s)
    currentValue = sTranspose @ w
    if (terminated == False):
        newStateTranspose = np.transpose(newState)
        newValue = newStateTranspose @ w
    else:
        newValue = 0

    return newState, reward, currentValue, newValue, newTerminated, newTruncated

def calc_td_error(reward, gamma, newValue, currentValue):    # Calculate squiggly thing (tdError)
    tdError = reward + (gamma * newValue) - currentValue
    return tdError

def update_critic(w, critic_step_size, tdError, state):    # Semi-grad update critic
    w += (critic_step_size * tdError * s)
    return w

def update_actor(state, a, Theta, actor_step_size, tdError, actor_discount):    # Policy grad update actor
    gradient = log_softmax_policy_gradient(s, a, Theta)
    Theta += (actor_step_size * tdError * actor_discount * gradient)
    return Theta
# Helper functions for actor-critic algorithm end here

