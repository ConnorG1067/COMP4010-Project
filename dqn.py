import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque


class ReplayMemory:

  def __init__(self, capacity=50000):
    self.buffer = deque(maxlen=capacity)

  def add(self, transition):
    self.buffer.append(transition)

  def sample(self, batch_size):
    transitions = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)
    return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones,dtype=np.float32))

  def __len__(self):
    return len(self.buffer)
  



class DoubleDQNAgent:
    def __init__(self, input_shape,num_actions, gamma = 0.90, epsilon = 1.0, epsilon_min= 0.05,epsilon_decay = 0.995, batch_size = 128, target_update_frequency=1000):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.steps_counter = 0


        #create q networks
        self.main_q_network = self._dqn_neural_network(input_shape, num_actions)
        self.target_q_network = self._dqn_neural_network(input_shape, num_actions)

        #set target network weights
        self.target_q_network.set_weights(self.main_q_network.get_weights())

        #create replay
        self.replay_memory = ReplayMemory()

    def _dqn_neural_network(self, input_dim, output_dim):
      model = keras.Sequential(
          [
              keras.Input(shape=(input_dim,)),
              keras.layers.Dense(64, activation="relu"),
              keras.layers.Dense(64, activation="relu"),
              keras.layers.Dense(output_dim, activation="linear"),
          ]
      )

      optimizer = keras.optimizers.Adam(learning_rate=0.0005)
      loss_function = keras.losses.Huber()
      model.compile(optimizer=optimizer, loss=loss_function)
      return model  


    def select_action(self, state):
      state = np.expand_dims(state, axis = 0)

      if np.random.rand() < self.epsilon:
        return np.random.randint(self.num_actions)
      else:
        q_values = self.main_q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def store_experience(self, transition):
      self.replay_memory.add(transition)

    @tf.function
    def train(self):
      if len(self.replay_memory) < 1000:
        return

      states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

      #update q values
      next_q_values = self.main_q_network(next_states, training = False)
      best_next_actions = keras.ops.argmax(next_q_values, axis=1)
      best_action_masks = keras.ops.one_hot(best_next_actions, self.num_actions)

      target_q_values_next = self.target_q_network(next_states, training = False)
      target_q_values = rewards + (1 - dones) * self.gamma * keras.ops.sum(keras.ops.multiply(target_q_values_next, best_action_masks), axis=1)

      
      mask = keras.ops.one_hot(actions, self.num_actions)
      with tf.GradientTape() as tape:
        current_q_values = self.main_q_network(states)
        current_q_values = keras.ops.sum(keras.ops.multiply(current_q_values, mask), axis=1)
        loss = self.main_q_network.loss(target_q_values, current_q_values)
      
      grads = tape.gradient(loss, self.main_q_network.trainable_variables)
      self.main_q_network.optimizer.apply_gradients(zip(grads, self.main_q_network.trainable_variables))


      #epsilon decay
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)      

      #update target network
      self.steps_counter += 1
      if self.steps_counter % self.target_update_frequency == 0:
        self.target_q_network.set_weights(self.main_q_network.get_weights())

    def save_model(self):
      self.main_q_network.save("./models/dqn/trained_double_dqn_model.keras")

    def load_model(self):
      self.main_q_network = keras.models.load_model("./models/dqn/trained_double_dqn_model.keras")
      self.target_q_network.set_weights(self.main_q_network.get_weights())
