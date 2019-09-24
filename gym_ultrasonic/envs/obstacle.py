import math
import random
import warnings

import numpy as np
from gym.spaces import Box
from shapely import affinity
from shapely import speedups
from shapely.geometry import Polygon, LineString
from typing import List

if not speedups.enabled:
    warnings.warn("Install Cython to enable shapely speedups")


class Obstacle:
    def __init__(self, position, width, height, angle=0):
        """
        Parameters
        ----------
        position: List[float]
            Obstacle position in world coordinates, (x, y).
        width: float
            Obstacle width.
        height: float
            Obstacle height.
        angle: float
            Obstacle rotation angle in degrees.
        """
        self.position = np.array(position, dtype=np.float32)
        self.width = width
        self.angle = angle
        self.height = height

    @property
    def polygon(self):
        """
        Returns
        -------
        Polygon
            Minimum rotated bounding box polygon.
        """
        coords = np.add(self.get_polygon_parallel_coords(), self.position)
        polygon_parallel = Polygon(coords)
        return affinity.rotate(polygon_parallel, self.angle)

    def get_polygon_parallel_coords(self):
        """
        Returns
        -------
        list
            Clock-wise vertices of a bounding box polygon, parallel to the world axes.
        """
        l, r, t, b = -self.width / 2, self.width / 2, self.height / 2, -self.height / 2
        return [(l, b), (l, t), (r, t), (r, b)]


class Robot(Obstacle):
    def __init__(self, position, width, height, angle=0):
        """
        Parameters
        ----------
        position: List[float]
            Robot's position in world coordinates, (x, y).
        width: float
            Robot's width.
        height: float
            Robot's height.
        angle: float
            Robot's rotation angle in degrees.
        """
        Obstacle.__init__(self, position, width, height, angle=angle)
        self.ultrasonic_sensor_angles = (-20, 0, 20)

    def move_forward(self, speed):
        """
        Parameters
        ----------
        speed: float
            Make a single step with this speed.
        """
        vec = np.multiply(self.direction_vector, speed)
        self.position += vec

    def turn(self, angle_step):
        """
        Parameters
        ----------
        angle_step: float
            CCW angle turn in degrees.
        """
        self.angle += angle_step

    def collision(self, obj):
        """
        Parameters
        ----------
        obj: Obstacle or List[Obstacle]
            An obstacle(s) to the for a collision.

        Returns
        -------
        bool
            If robot's minimum rotated bounding box collides with an obstacle(s).
        """
        if isinstance(obj, Obstacle):
            ret = self.polygon.intersects(obj.polygon)
            return ret
        else:
            # iterable
            return any(map(self.collision, obj))

    @property
    def direction_vector(self):
        """
        Returns
        -------
        np.ndarray
            Robot's direction vector, (x, y)
        """
        angle_rad = math.radians(self.angle)
        return np.array([np.cos(angle_rad), np.sin(angle_rad)])

    @property
    def sensor_position(self):
        """
        Returns
        -------
        np.ndarray
            Ultrasonic sensor position in world coordinates, (x, y)
        """
        return self.position + self.direction_vector * (self.width / 2)

    def ray_cast(self, obstacles: List[Obstacle], angle_target: float, max_dist=255):
        """
        Casts a ray at specific `angle_target` and checks if there an intersection with `obstacles`.

        Parameters
        ----------
        obstacles: List[Obstacle]
            List of obstacles in the scene.
        angle_target: float
            Sensor angle to ray cast.
        max_dist: float
            Max dist a sensor can capture.

        Returns
        -------
        min_dist: float
            Min dist to an obstacle, w.r.t. `angle_target`.
            If no obstacle is found at the ray intersection, `max_dist` is returned.
        intersection_xy: list or np.ndarray
            X and Y of the intersection point with an obstacle.
        """
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
        """
        Sets the current position to a random point from a `box`.

        Parameters
        ----------
        box: Box
            Allowed box space to sample a point from.
        """
        if box.shape != (2,):
            raise ValueError("Can sample a point from a plain 2D box only")
        self.position = box.sample()
        self.angle = random.randint(0, 360)