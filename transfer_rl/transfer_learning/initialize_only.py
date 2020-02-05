from .transfer_learner import TransferLearner

class TransferLearningInitializeOnly(TransferLearner):
    """
    This transfer learning only initialize the initial weights of the model, which should have been
    loaded earlier. All other function calls will just pass back to the caller.
    """

    def __init__(self, optim=None, models=None, total_frames=None, **kwargs):
        super(TransferLearningInitializeOnly, self).__init__(optim=optim, models=models, total_frame=total_frames)

    def update_learning_rates(self, cur_frame=0):
        """
        Learning rates don't change
        """
        pass

