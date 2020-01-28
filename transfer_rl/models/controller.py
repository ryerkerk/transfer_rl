import math


class Controller():
    """
    Parent class for controllers. No functionality, was meant to provide a template for controllers to follow

    """
    def __init__(self):
        pass

    def create_model(self, **kwargs):
        pass

    def sample_action(self, state):
        pass

    def train(self, memory, beta):
        pass

    def save_model(self, PATH):
        pass

    def load_model(self, PATH):
        pass

    def check_train(self, mem):
        pass
