import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # Filter out info messages from tensorflowt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_fully_connected_model(n_features, n_actions, hidden_layers=None,
                                 hidden_activation = 'relu',
                                 output_activation = 'linear'):
    """

    :param n_features: Number of input features to model
    :param n_actions: Number of output actions
    :param hidden_layers: Number of nodes in each hidden layer (default [10, 10, 10)
                          Number of hidden layers is equal to length of list
    :param hidden_activation:Activation function for hidden nodes (default 'relu')
    :param output_activation: Activation function for output nodes (default 'linear')
    :return:
    """
    if hidden_layers is None:       # Avoid mutable arguments
        hidden_layers = [10, 10, 10]

    model = Sequential()
    model.add(Dense(units=hidden_layers[0], input_shape=(n_features,), activation=hidden_activation))
    for h in hidden_layers[1:]:
        model.add(Dense(units=h, activation=hidden_activation))
    model.add(Dense(units=n_actions, activation=output_activation))

    return model