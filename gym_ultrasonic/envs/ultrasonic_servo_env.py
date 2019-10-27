import math
import random
from typing import List

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from shapely.geometry import LineString

from .obstacle import Robot, Obstacle, Servo
from .constants import WHEEL_VELOCITY_MAX, SERVO_ANGLE_MAX


class UltrasonicEnv(gym.Env):
    """
    A differential robot with one Ultrasonic sonar sensor, trying to avoid obstacles. Never stops (no target).
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    # dist to obstacles
    observation_space = spaces.Box(low=0., high=2000., shape=(1,))

    # wheels velocity, mm/s
    # action space box is slightly larger because of the additive noise
    action_space = spaces.Box(low=-WHEEL_VELOCITY_MAX, high=WHEEL_VELOCITY_MAX, shape=(2,))

    def __init__(self, n_obstacles=4, time_step=0.1):
        """
        Parameters
        ----------
        n_obstacles: int
            Num. of obstacles on the scene.
        """
        super().__init__()
        self.time_step = time_step
        self.scale_down = 5
        self.width = self.height = 3000 // self.scale_down

        # robot's position will be reset later on
        self.robot = Robot(width=115 / self.scale_down, height=140 / self.scale_down,
                           sensor_max_dist=self.observation_space.high[0])

        wall_size = 10
        indent = wall_size + max(self.robot.width, self.robot.height)
        self.allowed_space = spaces.Box(low=indent, high=self.width - indent, shape=(2,))

        walls = [
            Obstacle([0, self.height / 2], self.height, wall_size),  # left wall
            Obstacle([self.width, self.height / 2], self.height, wall_size),  # right wall
            Obstacle([self.width / 2, self.height], wall_size, self.width),  # top wall
            Obstacle([self.width / 2, 0], wall_size, self.width)  # bottom wall
        ]
        self.obstacles = []

        sample_obstacle_size = lambda: random.randint(int(0.75 * self.robot.width), self.robot.width * 5)

        for random_obstacle_id in range(n_obstacles):
            obst = Obstacle(position=random.sample(range(self.width), k=2),
                            width=sample_obstacle_size(),
                            height=sample_obstacle_size(),
                            angle=random.randint(0, 360))
            self.obstacles.append(obst)
        self.obstacles.extend(walls)

        self.state = self.init_state

        # rendering
        self.viewer = None
        self.ray_collision_transform = rendering.Transform()
        self.robot_transform = rendering.Transform()
        self.servo_transform = rendering.Transform()
        self._current_trajectory = None

        self.reset()

    def step(self, action):
        """
        Make a single step:
            1) move forward by `action[0]` mm;
            2) rotate by `action[1]` degrees;
            3) rotate servo by `action[2]` degrees (`UltrasonicServoEnv-v1` only).

        Parameters
        ----------
        action: List[float]
            Env action to perform.

        Returns
        -------
        observation: List[float]
            A list that contains one value - min dist to an obstacle on its way.
        reward: float
            A reward, obtained by taking this `action`.
        done: bool
            Whether the episode has ended.
        info: dict
            An empty dict. Unused.
        """
        vel_left, vel_right = action[:2]
        servo_turn = action[2] if len(action) == 3 else None
        move_step, robot_turn, trajectory = self.robot.diffdrive(vel_left, vel_right, sim_time=self.time_step)
        self._current_trajectory = trajectory
        self.robot.servo.rotate(servo_turn)
        reward, done = self.reward(trajectory, move_step, robot_turn, servo_turn)
        self.state = self.update_state()
        info = {}
        return self.state, reward, done, info

    def update_state(self):
        """
        Returns
        -------
        state: List[float]
            Min dist to obstacles.
        """
        min_dist, _ = self.robot.ray_cast(self.obstacles)
        return [min_dist]

    def reward(self, trajectory, move_step, angle_turn, servo_turn):
        """
        Computes the reward.

        Parameters
        ----------
        trajectory: LineString
            Trajectory of this step.
        move_step: float
            Move robot with `move_step` mm along its main axis.
        angle_turn: float
            Turn robot by `angle_turn` degrees.
        servo_turn: float
            Turn servo by `servo_turn` degrees, if not None.

        Returns
        -------
        reward: float
            A reward, obtained by applying this step.
        done: bool
            Whether the robot collided with any of obstacles.
        """
        if self.robot.collision(self.obstacles):
            return -1000, True
        for obstacle in self.obstacles:
            if obstacle.polygon.intersects(trajectory):
                return -1000, True
        reward = -2 + move_step - abs(angle_turn)
        if servo_turn is not None:
            reward -= abs(servo_turn)
        return reward, False

    @property
    def init_state(self):
        """
        Returns
        -------
        List[float]
            Initial env state (observation) which is the min dist to obstacles.
            It's not clear what the default "min dist to obstacles" is - 0, `sensor_max_dist` or a value in-between.
            But since we `update_state()` after each `reset()`, it should not matter.
        """
        return [0.]

    def reset(self):
        """
        Resets the state and spawns a new robot position.
        """
        self.state = self.init_state
        self.robot.reset(box=self.allowed_space)
        while self.robot.collision(self.obstacles):
            self.robot.reset(box=self.allowed_space)
        self.state = self.update_state()
        return self.state

    def init_view(self):
        """
        Initializes the Viewer (for displaying purpose only).
        """
        self.viewer = rendering.Viewer(self.width, self.height)
        robot_view = rendering.FilledPolygon(self.robot.get_polygon_parallel_coords())
        robot_view.add_attr(self.robot_transform)
        robot_view.set_color(r=0., g=0., b=0.9)

        # ultrasonic sensor collision circle
        circle_collision = rendering.make_circle(radius=7)
        circle_collision.add_attr(self.ray_collision_transform)
        circle_collision.set_color(1, 0, 0)
        self.viewer.add_geom(circle_collision)

        servo_view = rendering.FilledPolygon(self.robot.servo.get_polygon_parallel_coords())
        servo_view.add_attr(self.servo_transform)
        servo_view.add_attr(rendering.Transform(translation=(self.robot.servo_shift, 0)))
        servo_view.add_attr(self.robot_transform)
        servo_view.set_color(1, 0, 0)

        for obstacle in self.obstacles:
            polygon_coords = list(obstacle.polygon.boundary.coords)
            polygon = rendering.FilledPolygon(polygon_coords)
            self.viewer.add_geom(polygon)

        self.viewer.add_geom(robot_view)
        self.viewer.add_geom(servo_view)

    def render(self, mode='human'):
        """
        Renders the screen, robot, and obstacles.

        Parameters
        ----------
        mode: str
            The mode to render with.
            Either 'human' or 'rgb_array'.
        """
        if self.viewer is None:
            self.init_view()

        if self._current_trajectory is not None:
            trajectory_view = rendering.FilledPolygon(self._current_trajectory.boundary.coords)
            trajectory_view._color.vec4 = (0, 0, 0.9, 0.2)
            self.viewer.add_onetime(trajectory_view)

        _, intersection_xy = self.robot.ray_cast(self.obstacles)
        self.ray_collision_transform.set_translation(*intersection_xy)

        self.robot_transform.set_translation(*self.robot.position)
        self.robot_transform.set_rotation(math.radians(self.robot.angle))
        self.servo_transform.set_rotation(math.radians(self.robot.servo.angle))
        with_rgb = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=with_rgb)

    def __str__(self):
        return f"{super().__str__()}:\nobservation_space={self.observation_space.low, self.observation_space.high};" \
               f"\naction_space={self.action_space.low, self.action_space.high};" \
               f"\n{self.robot};" \
               f"\nnum. of obstacles: {len(self.obstacles) - 4}"  # 4 walls


class UltrasonicServoEnv(UltrasonicEnv):
    """
    A robot with one Ultrasonic sonar sensor and a servo that rotates the sonar.
    The task is the same: avoid obstacles. Never stops (no target).
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    # dist to obstacles, servo_angle
    observation_space = spaces.Box(low=np.array([UltrasonicEnv.observation_space.low[0], -SERVO_ANGLE_MAX]),
                                   high=np.array([UltrasonicEnv.observation_space.high[0], SERVO_ANGLE_MAX]))

    def __init__(self, n_obstacles=4, time_step=0.1, servo_angular_vel=120):
        """
        Parameters
        ----------
        n_obstacles: int
            Num. of obstacles on the scene.
        servo_angular_vel: float or str
            Servo angular velocity, degrees per sec.
        """
        super().__init__(n_obstacles=n_obstacles, time_step=time_step)
        if servo_angular_vel == 'learn':
            # vector move along main axis (mm), angle turn (degrees), servo turn (degrees)
            self.action_space = spaces.Box(low=-12, high=12, shape=(3,))
        _servo = self.robot.servo
        servo_angle_range = (self.observation_space.low[1], self.observation_space.high[1])
        self.robot.servo = Servo(_servo.width, _servo.height, angle_range=servo_angle_range,
                                 angular_vel=servo_angular_vel)

    @property
    def init_state(self):
        """
        Returns
        -------
        List[float]
            Initial env state (observation).
            It's not clear what the default "min dist to obstacles" is - 0, `sensor_max_dist` or a value in-between.
            But since we `update_state()` after each `reset()`, it should not matter.
        """
        return [0., 0.]

    def update_state(self):
        """
        Returns
        -------
        state: List[float]
            Min dist to obstacles and servo rotation angle.
        """
        state = super().update_state()
        state.append(self.robot.servo.angle)
        return state
