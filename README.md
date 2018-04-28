# lab_rl

A simple framework to easily make new agents for the Deepmind Lab environment. It's core benefits are:

- Possibility to change environment behaviour in Python (reduce action space, add dynamcis to actions, etc.)
- Easily implement new agents that integrate into the framework
- Generates detailed experiment documentation for reproduction
- Automatic stats logging for episodes
- Automatic model saving 
- Automatic creation of experiment plots

See a video of an trained agent here: https://youtu.be/1Q9pBl3YEmo 

## Acknowledgements

Thank you to [Google's Deepmind](https://deepmind.com/) for providing their experiment environment to the [public](https://github.com/deepmind/lab). 

The 'SimpleDQNAgent' and 'SimpleDQNModel' were inspired by <https://github.com/avdmitry/rl_3d>.

## Installation

### Install your system

This was build on **Ubuntu 16.04**, can't help for other systems but it shouldn't be too hard too adapt.

- Add proper GPU support by installing the right drivers and the [Nvidia Deep Learning software](https://developer.nvidia.com/deep-learning-software) like CUDA, CuDNN, CuBlas, etc.
- Add the following packages (needed for video rendering)

```ShellSession
sudo apt install qtbase5-dev libglu1-mesa-dev mesa-utils libglib2.0-0 ffmpeg libgtk2.0-dev pkg-config
```

### Install the original Lab environment

- For **Ubuntu 16.04** it's fairly simple to follow the installation instructions given in the original [build instructions](https://github.com/deepmind/lab/blob/master/docs/build.md).
- Then download the repo

```ShellSession
git clone git@github.com:deepmind/lab.git
cd lab
```

- Run the tests to see if it's working

### Install this framework

Basically all system software should be installed already, so just clone the repo

- Download the repo **inside** the Lab path

```ShellSession
git clone git@github.com:cowhi/lab_rl.git
```

- Very important! Modify the original BUILD instructions by adding the following to the end of the file:

```Text
py_binary(
    name = "launcher",
    srcs = ["lab_rl/launcher.py",
            "lab_rl/agents.py",
            "lab_rl/environments.py",
            "lab_rl/experiments.py",
            "lab_rl/models.py",
            "lab_rl/helper.py",
            "lab_rl/replaymemories.py"],
    data = [":deepmind_lab.so"],
    main = "lab_rl/launcher.py",
    )
```

- To run the maps we are using you can copy the whole **./lab_rl/lab_rl/** folder containing the lua files to the Lab environment located in **./assets/game_scripts/**. Then you can move the **.map**-files into the map folder **./assets/maps/**. That should be it.

### Prepare Python environment

Ok, here we are setting up the Python environment (not everything here is necessary to run the original lab only). I am using Anaconda to facilitate things.

- **The easy way**: Just use the [lab_rl.yml](../blob/master/conda/lab_rl.yml) file, which contains the packages I personally have installed. This might not work on your system without changes.

```ShellSession
conda env create -f conda/lab_rl.yml
source activate lab_rl
```

- **The slightly less easy way**: Create the environment from scratch. Should work the best for your particular system.

```ShellSession
conda create --name lab_rl python=2.7 tensorflow-gpu keras-gpu numpy matplotlib seaborn pip
source activate lab_rl
conda install -c mw gtk2
conda install -c menpo opencv3
```

## Running experiments

Run all commands from the **./lab/** path, not **./lab/lab_rl/**

- First you can try to see if you can load a simple agent, have it set up the logging structure, have it act in the environment for one episode, and display it as a video

```ShellSession
bazel run :launcher -- --play=True --show=True
```

- To run a full experiment which trains the 'SimpleDQNAgent' and generates a video after every epoch, just run

```ShellSession
bazel run :launcher -- --save_video=True
```

- You can set a lot of parameters for the experiments using the command line. Just check [launcher.py](../blob/master/launcher.py) to see your available options and add the variable too the command line call after the first `--` as `--variable=value`

## Logging

The launcher generates a logging path for every experiment at `~/.lab/$DATE$_$TIME$_$MAP$_$AGENT$/` containing the following:

- `args_dump.txt` - All experiment parameters in alphabetical order are saved here
- `experiment.log` - Logfile to see the progress of the environment with information about the training progress
- `stats_test.csv` - Logs all information about tests (every epoch)
- `stats_train.csv` - Logs all information about every training episode
- `models/` - Path were all the models are saved during training after every epoch
- `plots/` - Path to the plots generated during training to see the development of important experiment parameters
- `videos/` - Path to the videos generated during training (if set with `--save_video=True`) after every epoch

## Launcher

Initializes the experiment and deals with the commandline arguments, sets up the logging infrastructure, and starts the experiment.

## Experiments

This class describes all components of the experiment and coordinates training.

## Environments

All environments are based on the Environment class. For now all possible actions are only available as discrete actions. 

### LabAllActions

This class provides the agent with all available actions. 

### LabLimitedActions

This class provides the agent only with the ability to look left and right and move forward.


## Agents

All agents are based on the Agent class. This is the most important set of classes for developing new agents.

### DiscretizedRandomAgent

This agent just performs random steps. Use it only to test the environment using the `--play=True` option. Won't train anything.

### DummyAgent

This agent just performs a single action. Use it only to test the environment using the `--play=True` option. Won't train anything.

### SimpleDQNAgent

This agent resembles a very simplified version of the [original DQN](https://sites.google.com/a/deepmind.com/dqn/) implementation, but is sufficient enough to train on simple maps relatively fast and learn to solve given tasks. It is inspired by the [rl_3d](https://github.com/avdmitry/rl_3d) implementation.

## Models

This class defines the available models an agent can use during training.

### TensorflowModel

Baseclass for all Tensorflow models

### SimpleDQNModel

This class buils a simple model in Tensorflow. It consists of a neural network with an input layer respecting the image size that is given by the agent, 2 convolutional layers, 1 fully connected layer, and an output layer in the size of the available actions for the agent.

## ReplayMemories

This class defines the kind of replay memory an agent can use during training.

### SimpleReplayMemory

The most basic form of a replay memory. Just saves memories in a growing numpy array and selects random experiences during training.

## Generate new maps

This is a little bit annoying but I haven't found a quicker way:

- Copy one of the *.lua and *_run.lua in the lab_rl folder and give them a good name for your map and update the variables
- For the first run change the required factory to 'lab_rl.random_spawn_factory_map' in the NEW_run.lua file
- Copy the temporary new map to the lab_rl folder to keep it in the repo and to the assets/map/ folder to use it
- Change the required factory back to 'lab_rl.random_spawn_factory'

## TODOs

Please realize, this is a work in progress, so let me know what to improve. The following are already on my list (not sure I can do them all):

- Improve plots
- Make it easier to use (so people don't have to mess with existing files)
- Possible to use extra thread for plot and video generation (play and test) ?!
- Add support for continuous actions (discrete and continuous envs and agents)
- Add support for different model frameworks (pytorch!)
- Possible to use tensorboard??
- Document Matplotlib issue, change backend to 'Agg' in /home/ruben/anaconda3/envs/lab/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

## Support

If you like this and want to give me more freedom to work on it, I am happily accepting donations in BTC:

```Text
3BSdRNBFtPbMr5P1ApJZeNDqvKPczNvkKZ
```