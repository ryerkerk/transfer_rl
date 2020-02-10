import argparse

def check_bool(v):
    """
    Used to convert the "true" or "false" supplied in command line to boolean values.
    """
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Please set boolean value to true or false')

def parse_arg():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(usage="python %(prog)s --save_name SAVE_NAME [options]",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Study parameters
    parser.add_argument("--save_name", type=str, default="no-name", metavar=None,
                        help="File name for trained model and results")
    parser.add_argument("--total_frames", type=int, default=10e6,
                        help="Total number of frames (time steps) allowed during study")
    parser.add_argument("--render", type=check_bool, default=False,
                        help="Set to true to render model. No training occurs if model is rendered")

    # Environment parameters
    parser.add_argument("--env", type=str, default="Bipedal_Custom_Leg_Length-v0",
                        choices=['Bipedal_Custom_Leg_Length-v0'],
                        help="Name of openai gym environment")
    parser.add_argument("--leg_length", type=float, default=34,
                        help="Leg length of robot")
    parser.add_argument("--terrain_length_scale", type=int, default=2,
                        help="Values greater or smaller than 1 make terrain respectively longer or shorter")
    parser.add_argument("--fall_penalty", type=float, default=-50,
                        help="Penalty for falling over, should be negative value")
    parser.add_argument("--max_time_steps", type=int, default=1500,
                        help="Number of time steps allowed per environment run")

    # Network and model parameters
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on")
    parser.add_argument("--hidden_layers", type=str, default="[32,32,32,32]",
                        help="Number of nodes in each hidden layer")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=['ppo'],
                        help="Reinforcement learning algorithm")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'adagrad', 'sgd', 'rmsprop'],
                        help="Optimizer used with reinforcement model")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate of model")
    parser.add_argument("--gamma", type=float, default=0.98,
                        help="Reward discount rate")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="batch size for optimizer")
    parser.add_argument("--train_steps", type=int, default=80,
                        help="Number of training steps per batch")
    parser.add_argument("--action_std", type=float, default=0.5,
                        help="Standard deviation of noise applied to actions. Only used if adaptive action noise values are not defined")
    parser.add_argument("--action_std_start", type=float, default=-1,
                        help="Starting value for standard deviation of noise applied to actions, setting value will enable adaptive action noise")
    parser.add_argument("--action_std_final", type=float, default=-1,
                        help="Final value for standard deviation of noise applied to actions, setting value will enable adaptive action noise")
    parser.add_argument("--action_std_end", type=float, default=-1,
                        help="Proportion of frames over which to transition from initial adaptive noise value to final value")

    # Transfer learning parameters
    parser.add_argument("--initial_model", type=str, default='none',
                        help="Model to load and initialize new model through transfer learning. Model must be present in ./trained_models/")
    parser.add_argument("--transfer_learning", type=str, default="none",
                        choices=['none', 'initialize', 'freeze_all_but_first', 'freeze_then_thaw'],
                        help="Type of transfer learning, if applicable")
    parser.add_argument("--tl_end", type=float, default=0.25,
                        help="Parameter for certain types of transfer learning")
    parser.add_argument("--reset_final_layer", type=check_bool, default=False,
                        help="Reinitialize final layer of weights, only applies if initial model is specified")
    parser.add_argument("--add_noise_layers", type=int, default=0,
                        help="Add noise to this many layers, only applies if initial model is specified")
    parser.add_argument("--add_noise_alpha", type=float, default=1,
                        help="Amount of noise added is this value multiplied by standard deviation of noise in each layer")


    params = vars(parser.parse_args())

    # Get hidden layers from string to list
    params['hidden_layers'] = [int(x) for x in params['hidden_layers'].strip('[]').split(',')]

    return params