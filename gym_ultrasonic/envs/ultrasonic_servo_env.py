import math
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from shapely import affinity
from shapely.geometry import Polygon

from .obstacle import Robot, Obstacle


def filled_polygon_from_obstacle(obstacle: Obstacle):
    pos_rotated = list(obstacle.polygon.boundary.coords)
    return rendering.FilledPolygon(pos_rotated)


class UltrasonicServoEnv(gym.Env):
    metadata = {'render.modes': ['human'], 'video.frames_per_second': 1}

    # todo: specify reward_range

    def __init__(self):
        super().__init__()
        self.width = self.height = 600

        # robot's position will be reset later on
        self.robot = Robot([0, 0], width=40, height=25)

        wall_size = 5
        indent = wall_size + max(self.robot.width, self.robot.height)
        self.allowed_space = spaces.Box(low=indent, high=self.width - indent, shape=(2,))

        self.walls = [
            Obstacle([0, self.height / 2], wall_size, self.height),  # left wall
            Obstacle([self.width, self.height / 2], wall_size, self.height),  # right wall
            Obstacle([self.width / 2, self.height], self.width, wall_size),  # top wall
            Obstacle([self.width / 2, 0], self.width, wall_size)  # bottom wall
        ]
        self.obstacles = [
            Obstacle([500, 300], width=50, height=50),
            Obstacle([100, 200], width=35, height=35, angle=45),
            *self.walls
        ]
        self.action_space = spaces.Box(low=-3, high=3, shape=(2,))  # Forward/backward, left/right
        self.observation_space = spaces.Box(low=0, high=255, shape=(2,))
        self.state = None

        # rendering
        self.viewer = None
        self.cast_trans = []
        self.robot_trans = rendering.Transform()

        self.reset()

    def step(self, action):
        # multiply since output of neural net is [-1,1] and this would be too slow
        action = action * 3
        # todo move 3 to speed factor
        # Exectue continous actions
        self.robot.move_forward(speed=action[0])
        self.robot.turn(action[1])
        reward, done = self.reward(action)

        min, p, p = self.robot.singleUsSensors(self.obstacles)
        infrared = self.robot.infraredSensor(self.obstacles)

        self.state = [min, infrared]
        return np.copy(self.state), reward, done, {}

    def reward(self, action):
        if self.robot.collision(self.obstacles):
            return -500, True
        reward = -2 + action[0] - np.abs(action[1] / 2)
        return reward, False

    def reset(self):
        self.state = np.zeros(2)
        self.robot.reset(box=self.allowed_space)
        while self.robot.collision(self.obstacles):
            self.robot.reset(box=self.allowed_space)
        return np.copy(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.height)
            robot_view = rendering.FilledPolygon(self.robot.get_polygon_parallel_coords())
            robot_view.add_attr(self.robot_trans)

            for sensor_id in range(3):
                circle = rendering.make_circle(radius=5)
                cast_trans = rendering.Transform()
                self.cast_trans.append(cast_trans)
                circle.add_attr(cast_trans)
                circle.set_color(1, 0, 0)
                self.viewer.add_geom(circle)

            sensor_view = rendering.make_circle(3)
            sensor_view.add_attr(rendering.Transform(translation=(self.robot.width / 2, 0)))
            sensor_view.add_attr(self.robot_trans)
            sensor_view.set_color(1, 0, 0)

            for obj in self.obstacles:
                polygon = filled_polygon_from_obstacle(obj)
                self.viewer.add_geom(polygon)

            self.viewer.add_geom(robot_view)
            self.viewer.add_geom(sensor_view)

        min, points, pos = self.robot.usSensors(self.obstacles)
        self.cast_trans[0].set_translation(points[1][0], points[1][1])
        self.cast_trans[1].set_translation(points[0][0], points[0][1])
        self.cast_trans[2].set_translation(points[2][0], points[2][1])

        self.robot_trans.set_translation(*self.robot.position)
        self.robot_trans.set_rotation(math.radians(self.robot.angle))
        return self.viewer.render()
