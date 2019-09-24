import random
import warnings
import math

import numpy as np
from shapely import affinity
from shapely import speedups
from shapely.geometry import Polygon, LineString
from gym.spaces import Box
from typing import List

if not speedups.enabled:
    warnings.warn("Install Cython to enable shapely speedups")


class Obstacle:
    def __init__(self, position, width, height, angle=0):
        self.position = np.array(position, dtype=np.float32)
        self.width = width
        self.angle = angle
        self.height = height

    @property
    def polygon(self):
        coords = np.add(self.get_polygon_parallel_coords(), self.position)
        polygon_parallel = Polygon(coords)
        return affinity.rotate(polygon_parallel, self.angle)

    def get_polygon_parallel_coords(self):
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        return [(l, b), (l, t), (r, t), (r, b)]

    def get_position(self):
        return np.array(self.position)


class Robot(Obstacle):
    def __init__(self, position, width, height, angle=0):
        Obstacle.__init__(self, position, width, height, angle=angle)
        self.ultrasonic_sensor_angles = (-20, 0, 20)

    def move_forward(self, speed):
        vec = np.multiply(self.direction_vector, speed)
        self.position += vec

    def turn(self, value):
        self.angle += value / 2

    def collision(self, obj) -> bool:
        if isinstance(obj, Obstacle):
            ret = self.polygon.intersects(obj.polygon)
            return ret
        else:
            # iterable
            return any(map(self.collision, obj))

    @property
    def direction_vector(self):
        angle_rad = math.radians(self.angle)
        return np.array([np.cos(angle_rad), np.sin(angle_rad)])

    @property
    def sensor_position(self):
        """
        :return: Ultrasonic sensor position in world coordinates, (x, y)
        """
        return self.position + self.direction_vector * (self.width / 2)

    # returns distance to nearest object
    def rayCast(self, obstacles, angle_target=0, max_dist=255):
        # relative position of ultrasonic sensor
        sensor_pos = self.sensor_position
        angle_target = math.radians(self.angle + angle_target)
        target_direction = np.array([np.cos(angle_target), np.sin(angle_target)])
        ray_cast = LineString([sensor_pos, sensor_pos + target_direction * max_dist])
        min_dist = max_dist
        intersection_xy = [-max_dist, -max_dist]  # hide from a drawing screen

        for obj in obstacles:
            obj_pol = obj.polygon
            intersection = obj_pol.intersection(ray_cast)
            if isinstance(intersection, LineString):
                intersection_coords = np.array(intersection.coords)
                distances = np.linalg.norm(intersection_coords - sensor_pos, axis=1)
                argmin = distances.argmin()
                dist = distances[argmin]
                if dist < min_dist:
                    min_dist = dist
                    intersection_xy = intersection_coords[argmin]
        return min_dist, intersection_xy

    def reset(self, box: Box):
        if box.shape != (2,):
            raise ValueError("Can sample a point from a plain 2D box only")
        self.position = box.sample()
        self.angle = random.randint(0, 360)
