# Transfer Reinforcement Learning

This project explores transfer learning for reinforcement learning models. Products using reinforcement learning are likely to go through many design iterations during their life cycle. Eventually these changes will invalidate previously trained policies. Through transfer learning the previously trained policy may be able to reduce the costs of retraining the model for the new product design iterations.

This project uses the 2D walker environment available in the OpenAI gym, modified to accept the walker's leg length as an input parameter. Transfer learning demonstrated by training a policy for one walker, and then using that as the initial policy for a second walker of a different leg length.

![](https://raw.githubusercontent.com/ryerkerk/transfer_rl/master/images/sample_walkers.png)

## Requirements

- Python (3.6)
- Pytorch (1.14)
- gym (0.15.4)
- numpy (1.18.1)
- box2d-py (2.3.8)

A conda environment can be created using the provided .yml file.

`conda env update --file environment.yml --name transfer_rl`

Alternatively, the provided Dockerfile can be used to create an image of this project.

## Getting started

### Training from scratch
New studies can be run from the command line. At minimum a study name and walker leg length should be provided as command line arguments.

`python run.py --save_name=sample_34_leg_length --leg_length=34`

While the study is running the best performing policies, based on a running average reward, will be saved to the `trained_models/` directory. Once the study is complete a pickle file containing the results from each completed episode will be saved to the `results/` folder.

### Transfer learning
Transfer learning can be applied by also specifying an existing trained policy. This policy must exist as a .pt file in the trained_models folder, and the architectures of the existing policy and new policy must match (i.e., the `hidden_nodes` parameter). A sample policy, trained on a walker with 34 length legs, is provided in the `trained_models/` folder.

`python run.py --save_name=sample_transfer_learning --leg_length=30 --initial_model=leg_34_trained`

By default, if an `initial_model` is specified the new policy weights will be initialized to those of the `initial_model` and training will resume as normal. In my experimentation this was generally the simplest and most robust method of transfer learning, but other approaches are described in the following parameters section.

### Visualizing policies

Walkers can be visualized by setting the `render` parameter. Note that when the walker is visualized no training will occur.

`python run.py --save_name=sample_34_leg_length --leg_length=34 --render=True`

## Parameters

### Study parameters
- `--save_name` The name under which the trained model and results files will be saved.
- `--total_frames` The total number of frames, or time steps, that the study will be run for. (default=10e6)
- `--render` Set to true to visualize the walker. (default=False)

### Environment parameters
- `--env` Name of the environment in which to train the agent. Currently only `Bipedal_Custom_Leg_Length-v0` is implemented.
- `--leg_length` Length of the walkers legs. Both legs will be the same length. (default=34)
- `--terrain_length_scale` Setting this to values greater or less than 1 will respectively increase or decrease the size of the world the walker can traverse. (default=2)
- `--fall_penalty` The reward penalty that is applied should the walker fall over.
- `--max_time_steps` The maximum number of time steps that each episode is allowed to run.

### Network and model Parameters
- `--hidden_layers` This is a list of the number of nodes to include in each layer of the fully connected network. This parameters affects both the actor and critic networks. Changing the length of the list affects the number of hidden layers. (default = [32,32,32,32])
- `--algorithm` The reinforcement learning algorithm used to train the model, currently only 'ppo' is supported.
- `--optimizer` Optimization algorithm for updating the model parameters. (default='adam')
- `--learning_rate` Learning rate for opitmization algorithm. (default=3e-4)
- `--gamma` Reward discount rate. (default=0.98)
- `--batch_size` Number of frames to collect in memory buffer before training model. (default=10000)
- `--train_steps` Number of training steps to perform for each batch (default=80)
- `--action_std` The standard deviation of noise applied to the action space when sampling the model. (default = 0.5)
- `--action_std_start`, `--action_std_final`, and `--action_std_end` These three values can be set to use a scheduled action space noise. When the study because the standard deviation of the noise will be set equal to `action_std_start`. This value will then increase or decrease at a linear rate until it reaches `action_std_final`. `action_std_end` specifies the proportion of `total_frames` over which this transition will occur. If these values are not set then the value specified by `action_std` will be used for the entire study.
- `--device` Device on which to train, the code only currently works for training on a cpu.

### Transfer Learning Parameters
- `--initial_model` If an initial model is specified then it's parameters will be copied into the new model. (default='none')
- `--transfer_learning` Several approaches to transfer learning are available. `'none'` and `'initialize'` are equivalent, after the parameters are copied from the initial model training will proceed as usual. `'freeze_all_but_final'` will freeze the the parameters in all layers of the actor and critic models except for the final layer. `'freeze_then_thaw'` also freezes all layers except the final layer, but additional layers are then thawed and subject to training as the study progresses. (default='none')
- `--tl_end` The proportion of `total_frames` over which the model layers thaw and are subject to training. This value is only relevant if `'freeze_then_thaw'` transfer learning is applied. (default=0.25)
- `--reset_final_layer` If set to true this will reinitialize the parameters of the final model layer to random values. (default=false)
- `--add_noise_layers` This will add noise to the parameters in the final *n* layers of the model, where *n* is the value specified by this parameter. (default=0)
- `--add_noise_alpha` The amount of noise, relative to the standard deviation of the parameters in each layer, that will be applied. Only relevant if `add_noise_layers` is set to a value greater than 0. (default=1)
