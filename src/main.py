# #!/Users/dmitrijivanov/miniconda/bin/python3

import argparse
import sys
import os
from engine import engine_train, form_log_name

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def create_dir(save_path, model_name):
    model_directory = os.path.join(save_path, model_name)

    # Check if the directory already exists
    if os.path.exists(model_directory):
        # If directory exists, find the appropriate suffix number
        i = 1
        while True:
            suffixed_directory = f"{model_directory}_{i}"
            if not os.path.exists(suffixed_directory):
                break
            i += 1
        model_directory = suffixed_directory

    # Create the directory
    os.makedirs(model_directory)

    # Save the model or any other necessary files
    # For example, using torch.save() to save a PyTorch model:
    # torch.save(model.state_dict(), os.path.join(model_directory, "model.pth"))

    print(f'Models will be saved in: {model_directory}')

    return model_directory


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--tensorboard', action='store_true')

    parser.add_argument('-e', '--env', type=str, help='Environment')
    parser.add_argument('-n', '--n_actors', default=1, type=int, help='Number of actors')
    parser.add_argument('-nt', '--n_test_actors', default=5, type=int, help='Number of actors')

    parser.add_argument('--alg', default='sac', type=str, choices=['ppo', 'ddpg', 'sac', 'dqn'], help='alg')
    parser.add_argument('--res', action='store_true')  # resnet for dqn
    # parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('-p', '--save_path', default='./models/', type=str, help='path to save models')

    parser.add_argument('-al', '--actor_layers', default=2, type=int, help='Number of internal layers in Actor')
    parser.add_argument('-ald', '--actor_layer_dim', default=64, type=int, help='The dimension of internal Actor MLP layers')
    parser.add_argument('--tanh', action='store_true', help='Hyperbolic tan activation function')

    parser.add_argument('-cl', '--critic_layers', default=2, type=int, help='Number of internal layers in Critic')
    parser.add_argument('-cld', '--critic_layer_dim', default=64, type=int, help='The dimension of internal Critic MLP layers')

    parser.add_argument('--max_steps', default=0, type=int, help='max steps in episode')
    # МБ через эпизоды лучше
    # parser.add_argument('--episodes', default=20, type=int, help='total number of episodes')
    parser.add_argument('--total_steps', default=2e5, type=int, help='Number of eval episodes')

    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--pruning_start', default=0.2, type=float, help='Pruning start point (fractrion of the number of env steps)')
    parser.add_argument('--pruning_end', default=0.8, type=float, help='Pruning start point (fractrion of the number of env steps)')
    parser.add_argument('--pruning_iterations', default=2, type=int, help='Number of pruning iterations')
    # parser.add_argument('--pruning_speed', default=0.2, type=float, help='Fraction of weights pruned at each iteration')
    parser.add_argument('--target_sparsity', default=0.9, type=float, help='Fraction of weights pruned at each iteration')
    parser.add_argument('--er', action='store_true')

    parser.add_argument('-qw', '--quantize_weights', action='store_true')
    parser.add_argument('--quantize_activations', action='store_true')
    parser.add_argument('-qpc', '--quantize_per_channels', action='store_true', default=False, help='Quantize per channels or per tensors')
    parser.add_argument('-qs', '--quantize_start', default=0.0, type=float, help='Quantization start point (fraction of the number of env steps')
    parser.add_argument('-qe', '--quantize_end', default=0.0, type=float, help='Quantization end point (fraction of the number of env steps')

    parser.add_argument('-as', '--additional_steps', default=0, type=float, help='(in fraction) Additional steps (e.g. for quantization)')
    parser.add_argument('-asym', '--asymmetric', action='store_true')

    parser.add_argument('--n_eval_episodes', default=20, type=int, help='Number of eval episodes')
    parser.add_argument('--eval_freq', default=10000, type=int, help='Frequency of evaluating (in steps)')

    parser.add_argument('--video_record', action='store_true')
    parser.add_argument('--video_record_freq', default=100000, type=int)

    parser.add_argument('--buffer', default=1000000, type=int)
    # parser.add_argument('--buffer', default=10000, type=int)
    # orig - 2.5 * 10^4
    parser.add_argument('--exploration_fraction', default=0.01, type=float)

    parser.add_argument('--load', default='', type=str, help='Load pretrained weights')
    parser.add_argument('-lbnq', '--load_best_nq', action='store_true', default=False)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('-nql', '--not_quantize_last', action='store_true', default=False)

    # load weights before or apply to a trained NN
    parser.add_argument('--ptq', action='store_true', default=False)  # Post Training Quantization

    # Not forget to change for CNN DQN (хотя последний старт с 1e-4)
    parser.add_argument('--lr', default=1e-4, type=float)


    # parser.add_argument('--delta_threshold', default=0.001, type=float, help='delta threshold')
    # # parser.add_argument('--video_record', action='store_true')
    args = parser.parse_args(sys.argv[1:])

    return args


def main():
    args = parse_arguments()

    if args.no_train is not True:
        model_directory = create_dir(args.save_path, f'{form_log_name(args)}')
    else:
        model_directory = 'default'

    engine_train(model_directory, args)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
