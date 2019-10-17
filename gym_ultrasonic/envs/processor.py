from rl.core import Processor
from typing import List
import numpy as np


class NormalizeNonNegative(Processor):
    
    def __init__(self, sensor_max_dist, angle_range, action_scale):
        self.sensor_max_dist = sensor_max_dist
        self.angle_range = angle_range
        self.action_scale = action_scale
    
    def process_observation(self, observation):
        """
        Converts observations to `[0, 1]` range, suitable for spiking neural networks (not used here).

        Parameters
        ----------
        observation: List[float]
            Dist to obstacles, servo angle.
            Dist range: `[0, sensor_max_dist]` -> `[0, 1]`.
            Servo angle range: `angle_range` -> `[0, 1]`.

        Returns
        -------
        observation_norm: List[float]
            Normalized observation in range `[0, 1]`.
        """
        dist_norm = observation[0] / self.sensor_max_dist
        angle_min, angle_max = self.angle_range
        angle_norm = (observation[1] - angle_min) / (angle_max - angle_min)
        observation_norm = [dist_norm, angle_norm]
        return observation_norm

    def process_action(self, action_sigm):
        """
        Parameters
        ----------
        action_sigm: np.ndarray
            Actor network prediction from sigmoid activation function.
            Robot move step, robot turn. Both in range `[0, 1]`.

        Returns
        -------
        action_tanh: np.ndarray
            Scaled action in range `[-action_scale, action_scale]`.
        """
        action_tanh = (action_sigm - 0.5) * 2
        action_tanh *= self.action_scale
        return action_tanh
