import numpy as np
from keras.callbacks import Callback


class ExpandLogger(Callback):
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
        metrics = np.asarray(self.metrics)
        metrics = metrics[~np.isnan(metrics).any(axis=1)]
        if metrics.shape[0] > 0:
            logs['mean_q'] = metrics[:, mean_q_id].mean()
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


class DataDumpLogger(Callback):
    def __init__(self, fpath):
        super().__init__()
        self.fpath = fpath
        self.observations = []
        self.rewards = []
        self.actions = []

    def on_episode_begin(self, episode, logs=None):
        self.observations.append([])
        self.rewards.append([])
        self.actions.append([])

    def on_step_end(self, step, logs=None):
        self.observations[-1].append(logs['observation'])
        self.rewards[-1].append(logs['reward'])
        self.actions[-1].append(logs['action'])

    def on_train_end(self, logs=None):
        episode_id = []
        for episode, obs_episode in enumerate(self.observations):
            episode_id.append(np.repeat(episode, len(obs_episode)))
        episode_id = np.hstack(episode_id)
        observations = np.vstack(self.observations)
        rewards = np.hstack(self.rewards)
        actions = np.vstack(self.actions)
        observation_head = ['dist_to_obstacles', 'servo_angle']
        if observations.shape[1] > 2:
            observation_head.append('servo_turn')
        header = ['episode', *observation_head, 'robot_move', 'robot_turn', 'reward']
        data = np.c_[episode_id, observations, actions, rewards]
        np.savetxt(self.fpath, data, fmt='%.5f', delimiter=',', header=','.join(header), comments='')
