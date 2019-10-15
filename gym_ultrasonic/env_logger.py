import warnings

import numpy as np
from keras.callbacks import Callback


class UltrasonicLogger(Callback):
    """
    Acts as `BaseLogger` to expand logs with the average of observations, actions, mean_q, etc.
    """

    def __init__(self):
        super().__init__()
        self.observations = []
        self.rewards = []
        self.actions = []
        self.metrics = []
        self.metrics_names = []
        self.step = 0

    def on_train_begin(self, logs=None):
        self.metrics_names = self.model.metrics_names

    def on_episode_begin(self, episode, logs=None):
        """ Reset environment variables at beginning of each episode """
        self.observations = []
        self.rewards = []
        self.actions = []
        self.metrics = []

    def on_episode_end(self, episode, logs=None):
        """ Compute training statistics of the episode when done """
        mean_q_id = self.metrics_names.index('mean_q')
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            # first episode results in all nan values
            logs['mean_q'] = np.nanmean(self.metrics, axis=0)[mean_q_id]
        logs['reward_mean'] = np.mean(self.rewards)
        actions_mean = np.mean(self.actions, axis=0)
        logs['robot_move'] = actions_mean[0]
        logs['robot_turn'] = actions_mean[1]
        logs['dist_to_obstacles'] = np.mean(self.observations, axis=0)[1]
        del logs['nb_steps']  # don't show total num. of steps

    def on_step_end(self, step, logs=None):
        """ Update statistics of episode after each step """
        self.observations.append(logs['observation'])
        self.rewards.append(logs['reward'])
        self.actions.append(logs['action'])
        self.metrics.append(logs['metrics'])
        self.step += 1
