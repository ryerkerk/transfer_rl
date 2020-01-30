import argparse

def parse_arg():
    parser = argparse.ArgumentParser(usage="python %(prog)s --save_name SAVE_NAME [options]",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters
    # Switch save_name to required=True once finished testing
    parser.add_argument("--save_name", type=str, default="no-name", metavar=None,
                        help="File name for trained model and results")
    parser.add_argument("--env", type=str, default="Bipedal_Custom_Leg_Length-v0",
                        help="Name of openai gym environment")
    parser.add_argument("--terrain_length_scale", type=int, default=2,
                        help="Values greater or smaller than 1 make terrain respectively longer or shorter")
    parser.add_argument("--fall_penalty", type=float, default=-100,
                        help="Penalty for falling over, should be negative value")
    parser.add_argument("--torque_penalty", type=float, default=0.00035,
                        help="Penalty for applying torque, should be positive value")
    parser.add_argument("--head_balance_penalty", type=float, default=5,
                        help="Penalty for unbalanced head, should be positive value")
    parser.add_argument("--head_height_penalty", type=float, default=0,
                        help="Reward/Penalty for head moving up/down")
    parser.add_argument("--leg_sep_penalty", type=float, default=0,
                        help="Reward/Penalty for separation of upper legs. Meant to avoid doing the splits")
    parser.add_argument("--torque_diff_penalty", type=float, default=0,
                        help="Penalty for differences in torque powers each time step. Promotes smoother motion")

    parser.add_argument("--leg_length", type=int, default=34,
                        help="Leg length of robot")
    parser.add_argument("--initial_model", type=str, default='none',
                        help="Model to load and initialize new model through transfer learning. Model must be present in ./trained_models/")

    parser.add_argument("--max_time_steps", type=int, default=1500,
                        help="Number of time steps allowed per environment run")
    parser.add_argument("--total_frames", type=int, default=5e6,
                        help="Total number of frames (time steps) allowed during study")

    parser.add_argument("--hidden_layers", type=str, default="[32,32,32,32]",
                        help="Number of nodes in each hidden layer")

    parser.add_argument("--model", type=str, default="ppo",
                        help="Reinforcement model")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer used with reinforcement model")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate of model")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="batch size for optimizer")
    parser.add_argument("--train_steps", type=int, default=80,
                        help="Number of training steps per batch")

    parser.add_argument("--action_std", type=float, default=0.5,
                        help="Standard deviation of noise applied to actions")
    parser.add_argument("--action_std_decay", type=float, default=0.999,
                        help="Decay rate of noise (for ddpg only)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Reward discount rate")
    parser.add_argument("--render", type=bool, default=False,
                        help="Set to true to render model. No training occurs if model is rendered")

    params = vars(parser.parse_args())

    # Get hidden layers from string to list
    params['hidden_layers'] = [int(x) for x in params['hidden_layers'].strip('[]').split(',')]

    return params