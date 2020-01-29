import argparse

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Desc test")

    # Hyper-parameters
    parser.add_argument("--save_name", type=str, default="no_name")
    parser.add_argument("--env", type=str, default="Bipedal_Custom_Leg_Length-v0")
    parser.add_argument("--knee_contact_penalty", type=float, default=0.05)
    parser.add_argument("--terrain_length_scale", type=int, default=2)
    parser.add_argument("--leg_length", type=int, default=34)
    parser.add_argument("--initial_model", type=str, default='none')

    parser.add_argument("--max_time_steps", type=int, default=1500)
    parser.add_argument("--total_frames", type=int, default=1e7)

    parser.add_argument("--hidden_layers", type=str, default="[32, 32, 32, 32]")

    parser.add_argument("--model", type=str, default="ppo")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--train_steps", type=int, default=80)

    parser.add_argument("--action_std", type=float, default=0.5)
    parser.add_argument("--action_std_decay", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--render", type=bool, default=False)

    params = vars(parser.parse_args())

    # Get hidden layers from string to list
    params['hidden_layers'] = [int(x) for x in params['hidden_layers'].strip('[]').split(',')]

    return params