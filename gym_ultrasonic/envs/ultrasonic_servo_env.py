import math

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering

from .obstacle import Robot, Obstacle


class UltrasonicServoEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    # vector move along main axis (mm), angle turn (degrees)
    action_space = spaces.Box(low=-3, high=3, shape=(2,))

    # dist to obstacle
    observation_space = spaces.Box(low=0, high=255, shape=(1,))

    def __init__(self):
        super().__init__()
        self.width = self.height = 600

        # robot's position will be reset later on
        self.robot = Robot([0, 0], width=40, height=25)

        wall_size = 10
        indent = wall_size + max(self.robot.width, self.robot.height)
        self.allowed_space = spaces.Box(low=indent, high=self.width - indent, shape=(2,))

        walls = [
            Obstacle([0, self.height / 2], wall_size, self.height),  # left wall
            Obstacle([self.width, self.height / 2], wall_size, self.height),  # right wall
            Obstacle([self.width / 2, self.height], self.width, wall_size),  # top wall
            Obstacle([self.width / 2, 0], self.width, wall_size)  # bottom wall
        ]
        self.obstacles = [
            Obstacle([500, 300], width=50, height=50),
            Obstacle([100, 200], width=35, height=35, angle=45),
            *walls
        ]
        self.state = [0.]  # observation: min dist to obstacle

        # rendering
        self.viewer = None
        self.sensor_transforms = []
        self.robot_transform = rendering.Transform()

        self.reset()

    def step(self, action):
        """
        Make a single step of:
            1) moving forward with the speed `action[0]`;
            2) rotating by `action[1]` degrees.

        Parameters
        ----------
        action: List[float]
            Speed and angle actions for the next step.

        Returns
        -------
        observation: List[float]
            A list that contains one value - min dist to an obstacle, if any, on its way.
        reward: float
            A reward, obtained by taking this `action`.
        done: bool
            Whether the episode has ended.
        info: dict
            An empty dict. Unused.
        """
        move_step, angle_turn = action
        self.robot.move_forward(move_step)
        self.robot.turn(angle_turn)
        reward, done = self.reward(move_step, angle_turn)
        # central ultrasonic sensor distance
        min_dist, _ = self.robot.ray_cast(self.obstacles, angle_target=0)
        self.state = [min_dist]
        info = {}
        return self.state, reward, done, info

    def reward(self, speed_move, angle_turn):
        """
        Computes the reward.

        Parameters
        ----------
        speed_move: float
            Move with this `speed_move`.
        angle_turn: float
            Turn by this `angle_turn` degrees.

        Returns
        -------
        reward: float
            A reward, obtained by applying these steps.
        done: bool
            Whether the robot collided with an obstacle.rst.
        """
        if self.robot.collision(self.obstacles):
            return -500, True
        reward = -2 + speed_move - 3 * np.abs(angle_turn)
        return reward, False

    def reset(self):
        """
        Resets the state and spawns a new robot position.
        """
        self.state = [0.]
        self.robot.reset(box=self.allowed_space)
        while self.robot.collision(self.obstacles):
            self.robot.reset(box=self.allowed_space)
        return self.state

    def init_view(self):
        """
        Initializes the Viewer (for displaying purpose only).
        """
        self.viewer = rendering.Viewer(self.width, self.height)
        robot_view = rendering.FilledPolygon(self.robot.get_polygon_parallel_coords())
        robot_view.add_attr(self.robot_transform)

        for sensor_id_unused in range(len(self.robot.ultrasonic_sensor_angles)):
            circle = rendering.make_circle(radius=10)
            cast_trans = rendering.Transform()
            self.sensor_transforms.append(cast_trans)
            circle.add_attr(cast_trans)
            circle.set_color(1, 0, 0)
            self.viewer.add_geom(circle)

        sensor_view = rendering.make_circle(3)
        sensor_view.add_attr(rendering.Transform(translation=(self.robot.width / 2, 0)))
        sensor_view.add_attr(self.robot_transform)
        sensor_view.set_color(1, 0, 0)

        for obstacle in self.obstacles:
            polygon_coords = list(obstacle.polygon.boundary.coords)
            polygon = rendering.FilledPolygon(polygon_coords)
            self.viewer.add_geom(polygon)

        self.viewer.add_geom(robot_view)
        self.viewer.add_geom(sensor_view)

    def render(self, mode='human'):
        """
        Renders the screen, robot, and obstacles.

        Parameters
        ----------
        mode: str
            The mode to render with.
        """
        if self.viewer is None:
            self.init_view()

        for sensor_transform, sensor_angle in zip(self.sensor_transforms, self.robot.ultrasonic_sensor_angles):
            _, intersection_xy = self.robot.ray_cast(self.obstacles, angle_target=sensor_angle)
            sensor_transform.set_translation(*intersection_xy)

        self.robot_transform.set_translation(*self.robot.position)
        self.robot_transform.set_rotation(math.radians(self.robot.angle))
        with_rgb = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=with_rgb)
