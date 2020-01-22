import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # Filter out info messages from tensorflowt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from .network import create_fully_connected_model

def gaussian_likelihood(x, mu, log_std):
    """
    Copied from https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/core.py\

    """
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

class ppo():
    """
    Adapted from the TensorFlow 2.0 PPO implementation found at:
    https://github.com/jw1401/PPO-Tensorflow-2.0/tree/master/algorithms/ppo

    """

    def __init__(self, n_features, n_actions, hidden_layers=None,
                       hidden_activation = 'relu',
                       output_activation = 'linear'):
        self.actor = create_fully_connected_model(n_features, n_actions, hidden_layers, hidden_activation, output_activation)
        self.critic = create_fully_connected_model(n_features, n_actions, hidden_layers, hidden_activation, output_activation)

    def run(self, x):
        return self.actor.predict(x), self.critic.predict(x)

    def get_action(self, obs, std=0):
        mu, _ = self.run(obs)
        mu += tf.random.normal(tf.shape(mu)) * std
        mu = tf.clip_by_value(mu, -1, 1)

        return mu