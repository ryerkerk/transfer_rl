from .transfer_learner import TransferLearner


class TransferLearningFreezeThenThaw(TransferLearner):
    """
    This transfer learning function starts by freezing all but the final n layers. Then, as the study
    progresses, the frozen layers will one by one unfreeze and become subject to training. The tl_end
    argument gives the proportion of the study's total frames after which the entire model is subject
    to training.
    """

    def __init__(self, optim=None, models=None, total_frames=None, n_frozen_layers=1, tl_end=0.3):
        super(TransferLearningFreezeThenThaw, self).__init__(optim=optim, models=models, total_frame=total_frames)

        self.optim = optim

        self.tl_frame_end = tl_end*total_frames

        # Check lr defined in original optimizer
        self.lr = self.optim.param_groups[0]['lr']

        # Get rid of all parameter groups to start
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        # Loop over models (e.g., actor and critic)
        self.models = models
        for i in range(len(self.models)):
            layer_count = 0
            layers = list(self.models[i].named_children())
            # Loop over layers n model
            for _, cur_layer in layers[::-1]:
                # Only consider layers that define weights (i.e., ignore ReLu or Tanh layers)
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_frozen_layers:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})

        self.layer_count = layer_count


    def update_learning_rates(self, cur_frame=0):
        """
        This function updates which layers are frozen based on the current frame of the study.
        """

        # Calculate the number of layers that will not subject to training.
        n_layers = 1 + max(0, (self.layer_count-1) * cur_frame/self.tl_frame_end)

        # Get rid of all optimizer parameter groups
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        # Loop over models (e.g., actor and critic)
        for i in range(len(self.models)):
            layer_count = 0
            layers = list(self.models[i].named_children())
            # Loop over layers n model
            for _, cur_layer in layers[::-1]:
                # Only consider layers that define weights (i.e., ignore ReLu or Tanh layers)
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_layers:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})

