from typing import List

import numpy as np
from rl.core import Processor

from gym_ultrasonic.envs.constants import SERVO_ANGLE_MAX, SENSOR_DIST_MAX, WHEEL_VELOCITY_MAX


class UnitsProcessor(Processor):

    def __init__(self, n_sonars):
        self.n_sonars = n_sonars

    def process_observation(self, observation):
        """
        Converts observations to `[0, 1]` range, suitable for spiking neural networks (not used here).
        Processes the observation as obtained from the environment for use in an agent and returns it.

        Parameters
        ----------
        observation: List[float]
            Observation state list.
            `observation[0]` - dist to obstacles - `[0, sensor_max_dist]` -> `[0, 1]`.
            `observation[1]` - servo angle - `angle_range` -> `[0, 1]`. Not used in `UltrasonicEnv`.

        Returns
        -------
        observation_norm: List[float]
            Normalized `observation` in range `[0, 1]`.
        """
        distances = observation[:self.n_sonars]
        observation_norm = np.divide(distances, SENSOR_DIST_MAX)
        if len(observation) == self.n_sonars + 1:
            servo_angle_norm = (observation[-1] + SERVO_ANGLE_MAX) / (2 * SERVO_ANGLE_MAX)
            observation_norm = np.r_[observation_norm, servo_angle_norm]
        assert ((observation_norm >= 0.) & (observation_norm <= 1.)).all()
        return observation_norm.tolist()

    def process_action(self, action_tanh):
        """
        Parameters
        ----------
        action_tanh: np.ndarray
            Actor network prediction (last activation layer).
            All values are in range `[-1, 1]`.
            `action_sigm[0]`: robot move step;
            `action_sigm[1]`: robot turn angle;
            `action_sigm[2]`: servo turn angle (`UltrasonicServoEnv-v1`).

        Returns
        -------
        action_tanh: np.ndarray
            Scaled action in range `[-action_scale, action_scale]`.
        """
        action_tanh = action_tanh * WHEEL_VELOCITY_MAX
        return action_tanh
