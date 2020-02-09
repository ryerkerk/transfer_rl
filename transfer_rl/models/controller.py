import math


class Controller():
    """
    Parent class for controllers. No functionality, meant to provide a template for controllers to follow

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

    def set_adaptive_action_std(self, action_std_start, action_std_final, action_std_end, total_frames):
        """
        Initialize an adaptive action noise value variables, will only be set if
        adaptive noise values is used.

        :param action_std_start: Initial standard deviation of action noise
        :param action_std_final: Final standard deviation of action noise.
        :param action_std_end: Fraction of total frames for noise value to
        :param total_frames: Total frames in study
        """
        assert action_std_start >= 0, "Starting action std value needs to be greater than 0 when using adaptive action noise"
        assert action_std_final >= 0, "Ending action std value needs to be greater than 0 when using adaptive action noise"
        assert 0 <= action_std_end <= 1, "Transition period for adaptive action noise needs to be between 0 and 1"

        self.adaptive_action_std = True
        self.action_std_start = action_std_start
        self.action_std_final = action_std_final
        self.action_std_end_frame = action_std_end*total_frames
        self.action_std = self.action_std_start

    def update_action_std(self, frames):
        """
        Update the current action noise value (self.action_std) if necessary.

        If adaptive action noise is not used this function will do nothing.

        :param frames: Current number of frames run
        """

        if self.adaptive_action_std == True:
            # Calculate new noise value
            self.action_std = self.action_std_start \
                    + (self.action_std_final - self.action_std_start) * frames / self.action_std_end_frame
            if frames > self.action_std_end_frame:
                self.action_std = self.action_std_final
