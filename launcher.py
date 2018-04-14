from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import re
import time

from lab_rl.experiments import Experiment
from lab_rl.helper import create_dir, dump_args, prepare_logger, plot_experiment, summarize_runs

__author__ = "Ruben Glatt"
__copyright__ = "Ruben Glatt"
__license__ = "MIT"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Framework to guide RL in the Deepmind Lab environment")

    experiment_args = parser.add_argument_group('Experiment')
    experiment_args.add_argument('--runs', type=int, default=1,
                                 help='Number of runs to perform the experiment.')
    experiment_args.add_argument('--steps', type=int, default=200000,
                                 help='Number of steps to run the agent')
    experiment_args.add_argument('--log_level', type=str, default='info',
                                 help='The log level for the log file.')
    experiment_args.add_argument('--save_video', type=bool, default=False,
                                 help='If this is set a video is saved during testing.')
    experiment_args.add_argument('--show', type=bool, default=False,
                                 help='If this is set a video is shown during testing.')
    experiment_args.add_argument('--play', type=bool, default=False,
                                 help='If this is set the agent only runs some test steps.')
    experiment_args.add_argument('--backup_frequency', type=float, default=0.01,
                                 help='Frequency of model backups: backup_frequency * steps.')
    experiment_args.add_argument("--random_seed", type=int, default=123,
                                 help="Random seed for reproducible experiments.")


    environment_args = parser.add_argument_group('Environment')
    environment_args.add_argument('--env', type=str, default='LabLimitedActions',
                                  help='The environment class that we want to use.')
    environment_args.add_argument('--width', type=int, default=80,
                                  help='Horizontal size of the observations')
    environment_args.add_argument('--height', type=int, default=80,
                                  help='Vertical size of the observations')
    environment_args.add_argument('--fps', type=int, default=60,
                                  help='Number of frames per second')
    environment_args.add_argument('--runfiles_path', type=str, default=None,
                                  help='Set the runfiles path to find DeepMind Lab data')
    environment_args.add_argument('--level_script', type=str, default='lab_rl/apple_square_run',
                                  help='The environment level script to load')
    environment_args.add_argument('--color_channels', type=int, default=1,
                                  help='The number of color channels for the environment.')

    agent_args = parser.add_argument_group('Agent')
    agent_args.add_argument('--agent', type=str, default='DQNAgent',
                            help='The agent we want to use for training.')
    agent_args.add_argument('--frame_repeat', type=int, default=4,
                            help='The number of frames where an action is repeated.')
    agent_args.add_argument('--test_episodes', type=int, default=5,
                            help='The number of test episodes for evaluation.')
    agent_args.add_argument('--exploration_method', type=str, default='tau',
                            help='The way the agent performes exploration (tau/epsilon).')
    agent_args.add_argument('--epsilon_start', type=float, default=1.0,
                            help='Exploration rate (epsilon) at the beginning of training.')
    agent_args.add_argument('--epsilon_min', type=float, default=0.1,
                            help='Minimum value of exploration rate (epsilon) during training.')
    agent_args.add_argument('--epsilon_decay', type=float, default=0.8,
                            help='Percentage of all steps from starting epsilon to minimum epsilon.')
    agent_args.add_argument('--tau_start', type=float, default=100.0,
                            help='Temperature parameter (tau) for the softmax weighting.')
    agent_args.add_argument('--tau_decay', type=float, default=0.8,
                            help='Percentage of all steps for temperature parameter (tau) decay.')
    agent_args.add_argument('--tau_min', type=float, default=0.1,
                            help='Temperature parameter (tau) for the softmax weighting.')
    agent_args.add_argument('--step_penalty', type=float, default=0.001,
                            help='Penalty for performing any action.')

    model_args = parser.add_argument_group('Model')
    model_args.add_argument('--model', type=str, default='SimpleDQNModel',
                            help='The model we want to use for training.')
    model_args.add_argument('--load_model', type=str, default=None,
                            help='The path to a model to load for the agent.')
    model_args.add_argument('--alpha', type=float, default=0.001,
                            help='The learning rate (alpha) of the model.')
    model_args.add_argument('--gamma', type=float, default=0.99,
                            help='The discount factor (gamma) of the model.')
    model_args.add_argument('--input_width', type=int, default=80,
                            help='Horizontal size of the input images for the network.')
    model_args.add_argument('--input_height', type=int, default=80,
                            help='Vertical size of the input images for the network.')
    model_args.add_argument('--sequence_length', type=int, default=4,
                            help='Defines the number of images that are necessary to capture the dynamics.')
    model_args.add_argument('--batch_size', type=int, default=32,
                            help='Batch size during network training.')
    model_args.add_argument('--target_update_frequency', type=int,
                            default=2000,
                            help='Update frequency of the target network.')
    

    memory_args = parser.add_argument_group('Memory')
    model_args.add_argument('--memory', type=str, default='SimpleReplayMemory',
                            help='The replay memory we want to use for training.')
    memory_args.add_argument('--memory_size', type=int, default=50000,
                             help='Size of the replay memory.')
    memory_args.add_argument('--priority_factor', type=float, default=0.6,
                             help='Indicates how much prioritization is used for memory samples.')

    return parser.parse_args()


def make_path_structure(path_to_dir):
    print('Saving all in:', path_to_dir)
    paths = {
        'log_path': create_dir(path_to_dir),
        'model_path': create_dir(os.path.join(path_to_dir, 'models')),
        'video_path': create_dir(os.path.join(path_to_dir, 'videos')),
        'plot_path': create_dir(os.path.join(path_to_dir, 'plots'))}
    return paths


def main():
    # get commandline arguments
    args = parse_args()
    script = os.path.split(args.level_script.lower())
    level = re.sub('\_run$', '', script[-1])

    new_dir = "%s_%s_%s" % (
        str(time.strftime("%Y-%m-%d_%H-%M")),
        str(level),
        str(args.agent.lower()))
    # Plot path
    target_path = create_dir(os.path.join(os.path.expanduser("~"), ".lab", new_dir))
    plot_path = create_dir(os.path.join(target_path, 'plots'))
    # save arguments as a text file
    dump_args(target_path, args)

    for run in range(args.runs):
        print('###  RUN {num:02d}  #############################'.format(num=run))
        paths = {
            'log_path': None,
            'model_path': None,
            'video_path': None,
            'plot_path': None}
        
        # define and create log path
        path_to_dir = os.path.join(os.path.expanduser("~"), ".lab", new_dir, 'run_{num:02d}'.format(num=run))
        paths = make_path_structure(path_to_dir)
        if not args.play:
            # Initialize and start logger
            prepare_logger(paths['log_path'], args.log_level)
            _logger = logging.getLogger(__name__)
            _logger.info("Start")

        # Initialize and start experiment
        experiment = Experiment(args, paths)
        experiment.run()

        # Plot experiment
        if not args.play:
            # plot_experiment(paths['log_path'], 'stats_train', 'episode')
            # plot_experiment(paths['log_path'], 'stats_test', 'epoch')
            _logger.info("Finished")
    # Plot experiment
    if not args.play:
        if args.runs > 1:
            summarize_runs(target_path)
            plot_experiment(target_path, 'stats_train', 'episode')
            plot_experiment(target_path, 'stats_test', 'epoch')
    # TODO: get best model
    print('############# FINISHED ##############')


if __name__ == "__main__" and __package__ is None:
    main()
