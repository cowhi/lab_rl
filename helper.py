from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import errno
import glob
import logging
import os
import random
import scipy.stats
import shutil
import sys

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg')
import numpy as np
import pandas as pd


def argmax_tiebreaker(values):
    """ Gets a random index from all available indices with max value. """
    return random.choice(np.nonzero(values == np.amax(values))[0])


def calculate_stats(my_list, confidence=0.95):
    """ Returns  statistics about a list. """
    my_array = 1.0 * np.array(my_list)
    array_mean = np.mean(my_array)
    array_ste = scipy.stats.sem(my_array)
    array_std = my_array.std()
    conf_lower, conf_upper = scipy.stats.t.interval(confidence, len(my_array) - 1, loc=np.mean(my_array),
                                                    scale=scipy.stats.sem(my_array))
    return array_mean, array_ste, array_std, conf_lower, conf_upper


def copy_file(src, dest):
    _logger = logging.getLogger(__name__)
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        _logger.critical("Can't copy file - %s" % str(e))
        sys.exit(1)
    # eg. source or destination doesn't exist
    except IOError as e:
        _logger.critical("Can't copy file - %s" % str(e))
        sys.exit(1)


def create_dir(path_to_dir):
    _logger = logging.getLogger(__name__)
    if not os.path.isdir(path_to_dir):
        try:
            os.makedirs(path_to_dir)
        except Exception as e:
            _logger.critical("Can't create directory - %s (%s)" % (path_to_dir, str(e)))
            sys.exit(1)
    return path_to_dir


def delete_dir(path_to_dir):
    _logger = logging.getLogger(__name__)
    try:
        shutil.rmtree(path_to_dir)
    except Exception as e:
        _logger.critical("Can't delete directory - %s (%s)" % (path_to_dir, str(e)))
        sys.exit(1)


def dump_args(log_path, args):
    _logger = logging.getLogger(__name__)
    try:
        args_dump = open(os.path.join(log_path, 'args_dump.txt'), 'w', 0)
        args_dict = vars(args)
        for key in sorted(args_dict):
            args_dump.write("%s=%s\n" % (str(key), str(args_dict[key])))
        args_dump.flush()
        args_dump.close()
    except Exception as e:
        _logger.critical("Can't dump args - %s" % str(e))
        sys.exit(1)


def prepare_logger(log_path, log_level):
    # make sure no loggers are already active
    try:
        logging.root.handlers.pop()
    except IndexError:
        # if no logger exist the list will be empty and we
        # need to catch the resulting error
        pass
    # set the logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), None),
        format='[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s] %(message)s',
        filename=os.path.join(log_path, 'experiment.log'),
        filemode='w')


def print_stats(step, step_num, train_scores, elapsed_time):
    _logger = logging.getLogger(__name__)
    steps_per_s = 1.0 * step / elapsed_time
    steps_per_m = 60.0 * step / elapsed_time
    steps_per_h = 3600.0 * step / elapsed_time

    steps_remain = step_num - step
    remain_h = int(steps_remain / steps_per_h)
    remain_m = int((steps_remain - remain_h * steps_per_h) / steps_per_m)
    remain_s = int((steps_remain - remain_h * steps_per_h - remain_m * steps_per_m) / steps_per_s)
    elapsed_h = int(elapsed_time / 3600)
    elapsed_m = int((elapsed_time - elapsed_h * 3600) / 60)
    elapsed_s = int((elapsed_time - elapsed_h * 3600 - elapsed_m * 60))
    print("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s))
    _logger.info("{}% | Steps: {}/{}, {:.2f}M step/h, {:02}:{:02}:{:02}/{:02}:{:02}:{:02}".format(
        100.0 * step / step_num, step, step_num, steps_per_h / 1e6,
        elapsed_h, elapsed_m, elapsed_s, remain_h, remain_m, remain_s))
    mean_train = 0
    std_train = 0
    min_train = 0
    max_train = 0
    if len(train_scores) > 0:
        train_scores = np.array(train_scores)
        mean_train = train_scores.mean()
        std_train = train_scores.std()
        min_train = train_scores.min()
        max_train = train_scores.max()
    print("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train))
    _logger.info("Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
        len(train_scores), mean_train, std_train, min_train, max_train))


def plot_experiment(path_to_dir, file_name, search):
    df = pd.read_csv(os.path.join(path_to_dir, file_name + '.csv'))
    # print(df)
    for column in df.columns:
        plt.figure(figsize=(10, 4), dpi=80)
        plt.plot(df[search], df[column],
                 label=column, color='blue', linewidth=2.0)
        plt.ylabel(column, fontsize=20, fontweight='bold')
        plt.xlabel(search, fontsize=20, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(path_to_dir, 'plots', str(file_name) + '_' + str(column) + '.png'),
                    bbox_inches='tight')
        plt.close('all')


def get_human_readable(size, precision=2):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffix_index = 0
    while size > 1024 and suffix_index < 4:
        suffix_index += 1  # increment the index of the suffix
        size = size/1024.0  # apply the division
    return "%.*f%s" % (precision, size, suffixes[suffix_index])


def get_softmax(values, tau=1.0):
    """ Return softmax of the given values. """
    e = np.exp(np.array(values[0]) / tau)
    softmax = e / np.sum(e)
    # print(values[0], softmax)
    # workaround for numpy "sum not 1"-error = normalizing
    return softmax / sum(softmax)


def write_stats_file(path_to_file, *args):
    _logger = logging.getLogger(__name__)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    # creating new line for file
    line = ''
    for arg in args:
        if type(arg) is list:
            for elem in arg:
                line += str(elem) + ','
        else:
            line += str(arg) + ','
    line = line[:-1] + '\n'
    # write to file
    try:
        file_handle = os.open(path_to_file, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            with open(path_to_file, 'a+') as f:
                f.write(line)
        else:  # Something unexpected went wrong so re-raise the exception.
            _logger.critical("Can't write stats file - %s " % str(e))
            sys.exit()
    else:  # No exception, so the file must have been created successfully.
        with os.fdopen(file_handle, 'w') as file_obj:
            # Using `os.fdopen` converts the handle to an object that acts
            # like a regular Python file object, and the `with` context
            # manager means the file will be automatically closed when
            # we're done with it.
            file_obj.write(line)


def summarize_runs(path_to_dir):
    _logger = logging.getLogger(__name__)
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/run_*/')

    for kind in ['train', 'test']:
        run_files = [os.path.join(run_dir, 'stats_' + kind + '.csv')
                     for run_dir in run_dirs]
        df = pd.concat((pd.read_csv(run_file) for run_file in run_files))
        if kind == 'train':
            interval = 'episode'
            step_term = 'steps'
            reward_term = 'reward'
        elif kind == 'test':
            interval = 'epoch'
            step_term = 'steps_avg'
            reward_term = 'reward_avg'
        else:
            sys.exit(1)
        steps = df.groupby(interval)[step_term]
        steps = list(steps)
        reward = df.groupby([interval])[reward_term]
        reward = list(reward)
        summary = []
        for episode in range(0, len(reward)):
            step_mean, _, _, _, _ = \
                calculate_stats(steps[episode][1])
            reward_mean, _, _, _, _ = \
                calculate_stats(reward[episode][1])
            summary.append([int(steps[episode][0]),
                            step_mean,
                            reward_mean])
        header = [interval, 'steps_mean', 'reward_mean']
        try:
            with open(os.path.join(path_to_dir, 'stats_' + kind + '.csv'), 'w') \
                    as csvfile:
                writer = csv.writer(csvfile,
                                    dialect='excel',
                                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(header)
                for data in summary:
                    writer.writerow(data)
        except IOError as e:
            _logger.critical("Can't write stats file - %s " % str(e))
            sys.exit()
