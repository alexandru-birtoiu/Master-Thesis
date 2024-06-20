# Introduction

Welcome to the "How many eyes does a robot need when learning new tasks?"
master thesis repository! This repository is dedicated to studying the optimal camera setup for visuomotor learning using behavioral cloning. It includes methods for creating datasets for four different tasks, training various network architectures, and benchmarking various camera setups. Additionally, it provides tools for evaluating the models by using them to predict the movement of the robot in simulation.

## Table of Contents

- [Setup Environment](#setup-environment)
- [Configuration Settings](#configuration-settings)
- [Dataset Creation](#dataset-creation)
- [Training a Model](#training-the-model)
- [Testing a Model](#testing-the-model)
- [Evaluating a Model](#evaluating-the-model)


## Setup Environment

### Create a virtual environment:

```sh
python3 -m venv venv
```

### Activate virtual environment
```sh
source venv/bin/activate
```

### Install the required packages:
```sh
pip install -r requirements.txt
```



## Configuration Settings

The configuration file `config.py` allows you to set various parameters for training, evaluation, and data gathering. Here are some key settings you can adjust:

### Model and Task Types:

- **MODEL_TYPE**: Type of camera setup (EGO_AND_BIRDSEYE, BIRDSEYE_MULTIPLE, EGOCENTRIC, BIRDSEYE).
- **TASK_TYPE**: The current task (CUBE_TABLE, INSERT_CUBE, CUBE_DEPTH, TEDDY_BEAR).
- **PREDICTION_TYPE**: Type of robot control prediction (VELOCITY, POSITION, TARGET_POSITION).
- **IMAGE_TYPE**: Type of image used (RGB, D, RGBD).
- **IMAGE_SIZE_TRAIN**: The image size currently used, both for training the model and then inference (128, 84, 64, 32).

### Training Parameters:

- **EPOCHS_TO_TRAIN**: Number of epochs to train.
- **LEARNING_RATE**: Learning rate for the optimizer.
- **BATCH_SIZE**: Size of the training batches.
- **USE_LSTM**: Use LSTM layers (True/False).
- **USE_TRANSFORMERS**: Use transformer architecture (True/False).

### Data Gathering:

- **NO_EPISODES**: Number of expert demosntrations


## Dataset Creation

To create a dataset for a specific task, you need to set the `NO_EPISODES` and the `TASK_TYPE` in the configuration file. Run the following script to create the dataset:

```sh
python create_dataset.py
```
This script will automatically create the folders `images` and `labels` based on the current configuration.



## Training a Model

You can train any of the three types of network architectures provided:

- `network_base.py`
- `network_transformers.py`
- `network_transformers_lstm.py`

Ensure that the relevant settings such as epochs to train, image type, current camera setups, learning rate, and whether to resume training are correctly configured in the `config.py` file. Moreover for the base network both `USE_TRANSFORMERS` and `USE_LSTM` should be turned off, while for the transformers one, only the `USE_TRANSFORMERS` should be enabled. Then, you can run the training script:

```sh
python {model_file}
```

Replace `{model_file}` with the name of the chosen model type, one of the three listed above.

All of these scripts will automatically create the `models` and `loss_figs` folder, and save the model at each epoch into them.



## Testing a Model

After training, you can test a specific epoch by setting the `EPOCH` in the configuration file and running the test script:

```sh
python test_trained_model.py
```

This will test the model for one episode without performing any automatic evaluation.


## Evaluating a Model

To evaluate a model, use the evaluation scripts specific to the task. For example, to evaluate the `INSERT_CUBE` task, run:

```sh
python evaluate_trained_model_insert.py
```
This script will evaluate all the epochs from 1 to the specified `EPOCH` and return the performance specific to that task over 50 episodes.
