class TransferLearner():
    """
    Parent class for transfer learning models.

    """
    def __init__(self, optim=None, models=None, total_frames=None, **kwargs):
        self.optim = optim
        self.models = models
        self.total_frames = total_frames
        pass

    def update_learning_rates(self, cur_frame=0):
        pass