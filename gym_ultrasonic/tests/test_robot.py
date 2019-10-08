import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from shapely.geometry import Polygon

from gym_ultrasonic.envs.obstacle import Robot, Obstacle


class TestRobotMethods(unittest.TestCase):

    def setUp(self):
        self.robot = Robot([300, 300], 50, 30)

    def test_turn(self):
        curr_angle = self.robot.angle
        angle_turn = 20
        self.robot.turn(angle_turn)
        self.assertAlmostEqual(self.robot.angle, curr_angle + angle_turn)

    def test_move_forward_straight(self):
        self.robot.angle = 0
        pos = np.copy(self.robot.position)
        self.robot.move_forward(move_step=1)
        pos[0] = pos[0] + 1
        assert_array_almost_equal(self.robot.position, pos)

    def test_move_forward_angle(self):
        self.robot.angle = 45
        pos = np.copy(self.robot.position)
        self.robot.move_forward(move_step=1)
        increment = 0.5 ** 0.5
        pos += increment
        assert_array_almost_equal(self.robot.position, pos)

    def test_collision(self):
        obstacle = Obstacle([350, 350], 150, 150)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_on_line(self):
        self.robot = Robot([300, 300], 100, 100)
        obstacle = Obstacle([400, 300], 100, 100)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_on_line_slim(self):
        self.robot = Robot([300, 300], 100, 50)
        obstacle = Obstacle([400, 300], 100, 100)
        self.assertTrue(self.robot.collision(obstacle))

    def test_collision_rotation(self):
        self.robot = Robot([300, 300], 100, 100)
        self.robot.angle = 46
        obstacle = Obstacle([350, 350], 150, 150)
        self.assertTrue(self.robot.collision(obstacle))

    def test__edge_collision_rotation(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 45
        obstacle = Obstacle([350, 350], 100, 50)
        self.assertTrue(self.robot.collision(obstacle))

    def test_no_collision(self):
        self.robot = Robot([300, 300], 50, 30)
        obstacle = Obstacle([500, 500], 30, 30)
        self.assertFalse(self.robot.collision(obstacle))

    def test_ray_casting(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 0
        obstacle = Obstacle([400, 300], 50, 50)
        min_dist, p_xy = self.robot.ray_cast([obstacle], angle_target=0)
        self.assertEqual(min_dist, 50.0)
        assert_array_almost_equal(p_xy, [375, 300])

    def test_ray_casting_nohit(self):
        self.robot = Robot([300, 300], 50, 50)
        self.robot.angle = 0
        obstacle = Obstacle([300, 400], 50, 50)
        min_dist, _ = self.robot.ray_cast([obstacle], angle_target=0)
        self.assertEqual(min_dist, self.robot.sensor_max_dist)

    def test_coll_rotated(self):
        v1 = [[425.643330675622, 247.83978898773091], [425.643330675622, 277.83978898773091], [
            475.643330675622, 277.83978898773091], [475.643330675622, 247.83978898773091]]
        v2 = [[475, 275], [475, 325], [525, 325], [525, 275]]
        p1 = Polygon(v1)
        p2 = Polygon(v2)
        self.assertTrue(p1.intersects(p2))


if __name__ == '__main__':
    unittest.main()
