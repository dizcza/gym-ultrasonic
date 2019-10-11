import math
import random
import unittest

import gym
import numpy as np
from numpy.testing import assert_array_almost_equal

import gym_ultrasonic


class TestUltrasonicEnv(unittest.TestCase):

    def setUp(self):
        self.env = gym.make('UltrasonicServo-v0')
        self.env.robot.speed = 1
        self.env.reset()
        # self.env.robot.position = np.divide([self.env.width, self.env.height], 2.)
        # self.env.robot.angle = 0

    def tearDown(self):
        self.env.close()

    def test_reset(self):
        self.env.reset()
        assert_array_almost_equal(self.env.state, [0.])
        self.assertFalse(self.env.robot.collision(self.env.obstacles))

    def test_step_do_nothing(self):
        robot_pos = np.copy(self.env.robot.position)
        robot_angle = self.env.robot.angle
        speed, angle_turn = 0, 0
        _, _, done, _ = self.env.step(action=(speed, angle_turn))
        self.assertFalse(done)
        assert_array_almost_equal(robot_pos, self.env.robot.position)
        self.assertEqual(robot_angle, self.env.robot.angle)
        self.assertFalse(self.env.robot.collision(self.env.obstacles))

    def test_step_rotate_360(self):
        speed = 0
        for angle_turn in (180, 360):
            with self.subTest(angle_turn=angle_turn):
                _, _, done, _ = self.env.step(action=(speed, angle_turn))
                self.assertFalse(done)
                self.assertFalse(self.env.robot.collision(self.env.obstacles))

    def test_step_collide_any(self):
        obstacle = self.env.obstacles[0]
        vec_to_obstacle = obstacle.position - self.env.robot.position
        vx, vy = vec_to_obstacle
        angle_target = np.arctan2(vy, vx)
        if angle_target < 0:
            angle_target += 2 * math.pi
        angle_target = math.degrees(angle_target)
        angle_turn = angle_target - self.env.robot.angle
        self.env.robot.turn(angle_turn)
        dist_to_obstacle = np.linalg.norm(vec_to_obstacle)
        min_dist, reward, done, _ = self.env.step(action=(dist_to_obstacle, 0))
        assert_array_almost_equal(min_dist, [0.], decimal=4)
        self.assertTrue(reward < 0)
        self.assertTrue(done)

    def test_step_collide_towards(self):
        dist_to_obstacle, _ = self.env.robot.ray_cast(self.env.obstacles, angle_target=0)
        min_dist, reward, done, _ = self.env.step(action=(dist_to_obstacle, 0))
        assert_array_almost_equal(min_dist, [0.], decimal=4)
        self.assertTrue(reward < 0)
        self.assertTrue(done)

    def test_large_robot(self):
        """
        A robot is so large - immediate collision.
        """
        self.env.robot.width = self.env.width
        _, _, done, _ = self.env.step(action=(0, 0))
        self.assertTrue(done)

    def test_render(self):
        self.env.render()


if __name__ == '__main__':
    random.seed(27)
    np.random.seed(27)
    unittest.main()
