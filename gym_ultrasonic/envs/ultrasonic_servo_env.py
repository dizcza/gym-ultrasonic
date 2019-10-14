import math
import random
from typing import List

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
    observation_space = spaces.Box(low=np.array([-90, 0]), high=np.array([90, 2000]))

    def __init__(self, servo_angular_vel=30):
        """
        Parameters
        ----------
        servo_angular_vel: float
            Servo angular velocity, degrees per sec.
        """
        super().__init__()
        self.scale_down = 5
        self.width = self.height = 3000 // self.scale_down

        servo_angle_range = (self.observation_space.low[0], self.observation_space.high[0])

        # robot's position will be reset later on
        self.robot = Robot(width=120 / self.scale_down, height=90 / self.scale_down, speed=3,
                           sensor_max_dist=self.observation_space.high[1],
                           servo_angle_range=servo_angle_range,
                           servo_angular_vel=servo_angular_vel)

        wall_size = 10
        indent = wall_size + max(self.robot.width, self.robot.height)
        self.allowed_space = spaces.Box(low=indent, high=self.width - indent, shape=(2,))

        walls = [
            Obstacle([0, self.height / 2], wall_size, self.height),  # left wall
            Obstacle([self.width, self.height / 2], wall_size, self.height),  # right wall
            Obstacle([self.width / 2, self.height], self.width, wall_size),  # top wall
            Obstacle([self.width / 2, 0], self.width, wall_size)  # bottom wall
        ]
        self.obstacles = []

        sample_obstacle_size = lambda: random.randint(self.robot.height // 2, self.robot.width * 3)

        for random_obstacle_id in range(5):
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
        self.robot.servo.step_rotate()
        reward, done = self.reward(move_step, angle_turn)
        self.update_state()
        info = {}
        return self.state, reward, done, info

    def update_state(self):
        """
        Updates the env state which is a list, containing two values: min dist to obstacles and servo rotation angle.
        """
        min_dist, _ = self.robot.ray_cast(self.obstacles)
        self.state = [min_dist, self.robot.servo.angle]

    def reward(self, move_step, angle_turn):
        """
        Computes the reward.

        Parameters
        ----------
        move_step: float
            Move robot with `move_step` mm along its main axis.
        angle_turn: float
            Turn robot by `angle_turn` degrees.

        Returns
        -------
        reward: float
            A reward, obtained by applying this step.
        done: bool
            Whether the robot collided with any of obstacles.
        """
        if self.robot.collision(self.obstacles):
            return -500, True
        reward = -2 + move_step - 3 * np.abs(angle_turn)
        return reward, False

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
        return [self.robot.sensor_max_dist, 0.]

    def reset(self):
        """
        Resets the state and spawns a new robot position.
        """
        self.state = self.init_state
        self.robot.reset(box=self.allowed_space)
        while self.robot.collision(self.obstacles):
            self.robot.reset(box=self.allowed_space)
        self.update_state()
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

        _, intersection_xy = self.robot.ray_cast(self.obstacles)
        self.ray_collision_transform.set_translation(*intersection_xy)

        self.robot_transform.set_translation(*self.robot.position)
        self.robot_transform.set_rotation(math.radians(self.robot.angle))
        self.servo_transform.set_rotation(math.radians(self.robot.servo.angle))
        with_rgb = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=with_rgb)
