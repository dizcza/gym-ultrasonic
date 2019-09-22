import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering

from .obstacles import Robot, Obstacle


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
        self.speed = 0.5
        self.action_space = spaces.Box(low=-3, high=3, shape=(2,))  # Forward/backward, left/right
        self.observation_space = spaces.Box(low=0, high=255, shape=(2,))

        self.viewer = None
        self.state = None

        self.reset()

    def step(self, action):
        # multiply since output of neural net is [-1,1] and this would be too slow
        action = action * 3
        # todo move 3 to speed factor
        # Exectue continous actions
        self.robot.move_forward_speed(action[0])
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

            robot = rendering.FilledPolygon(self.robot.get_drawing())
            c1 = rendering.make_circle(2)
            c2 = rendering.make_circle(2)
            c3 = rendering.make_circle(2)
            start = rendering.make_circle(3)
            obs = rendering.FilledPolygon(self.obstacles[0].get_drawing())
            obs2 = rendering.FilledPolygon(self.obstacles[1].get_drawing())
            for wall in self.walls:
                draw = rendering.FilledPolygon(wall.get_drawing_static_position())
                # draw.set_color(0,0,1)
                self.viewer.add_geom(draw)
            self.obtrans = rendering.Transform()
            self.obtrans2 = rendering.Transform()
            self.casttrans = rendering.Transform()
            self.casttrans2 = rendering.Transform()
            self.casttrans3 = rendering.Transform()
            self.starttrans = rendering.Transform()
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            obs.add_attr(self.obtrans)
            obs2.add_attr(self.obtrans2)
            c1.add_attr(self.casttrans)
            c2.add_attr(self.casttrans2)
            c3.add_attr(self.casttrans3)
            start.add_attr(self.starttrans)
            c1.set_color(1, 0, 0)
            c2.set_color(1, 0, 0)
            c3.set_color(1, 0, 0)
            start.set_color(1, 0, 0)
            self.viewer.add_geom(robot)
            self.viewer.add_geom(start)
            self.viewer.add_geom(obs)
            self.viewer.add_geom(obs2)
            self.viewer.add_geom(c1)
            self.viewer.add_geom(c2)
            self.viewer.add_geom(c3)

        min, points, pos = self.robot.usSensors(self.obstacles)

        x, y = self.obstacles[0].get_position(
        )[0], self.obstacles[0].get_position()[1]
        self.obtrans.set_translation(x, y)
        x, y = self.obstacles[1].get_position(
        )[0], self.obstacles[1].get_position()[1]
        self.obtrans2.set_translation(x, y)
        self.obtrans2.set_rotation(self.obstacles[1].angle * np.pi / 180)
        x = self.robot.get_position()[0]
        y = self.robot.get_position()[1]
        rot = self.robot.angle
        self.starttrans.set_translation(pos[0], pos[1])
        self.casttrans.set_translation(points[1][0], points[1][1])
        self.casttrans2.set_translation(points[0][0], points[0][1])
        self.casttrans3.set_translation(points[2][0], points[2][1])

        self.robottrans.set_translation(x, y)
        self.robottrans.set_rotation(rot * np.pi / 180)
        return self.viewer.render()
