from typing import List, Tuple

import numpy as np
from rl.core import Processor


class NormalizeNonNegative(Processor):
    
    def __init__(self, sensor_max_dist, action_scale, angle_range):
        """
        Parameters
        ----------
        sensor_max_dist: float
            Max Ultrasonic sensor dist.
        action_scale: Union[float, List[float]]
            Scale actor network output before feeding it in the env.
        angle_range: Tuple[float]
            Servo angle range.
            `None` for `UltrasonicEnv`.
        """
        self.sensor_max_dist = sensor_max_dist
        self.action_scale = action_scale
        self.angle_range = angle_range
    
    def process_observation(self, observation):
        """
        Converts observations to `[0, 1]` range, suitable for spiking neural networks (not used here).

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
        dist_norm = observation[0] / self.sensor_max_dist
        observation_norm = [dist_norm]
        if len(observation) > 1:
            assert self.angle_range is not None, "Make sure you use `UltrasonicServoEnv`"
            angle_min, angle_max = self.angle_range
            angle_norm = (observation[1] - angle_min) / (angle_max - angle_min)
            observation_norm.append(angle_norm)
        return observation_norm

    def process_action(self, action_sigm):
        """
        Parameters
        ----------
        action_sigm: np.ndarray
            Actor network prediction from sigmoid activation function.
            All values are in range `[0, 1]`.
            `action_sigm[0]`: robot move step;
            `action_sigm[1]`: robot turn angle;
            `action_sigm[2]`: servo turn angle (`UltrasonicServoEnv-v1`).

        Returns
        -------
        action_tanh: np.ndarray
            Scaled action in range `[-action_scale, action_scale]`.
        """
        action_tanh = (action_sigm - 0.5) * 2
        action_tanh *= self.action_scale
        return action_tanh
