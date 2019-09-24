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
        self.ultrasonic_sensor_angles = [-20, 0, 20]

    def move_forward(self, speed):
        angle_rad = math.radians(self.angle)
        vec = [np.cos(angle_rad), np.sin(angle_rad)]
        vec = np.multiply(vec, speed)
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

    def infraredSensor(self, obstacles: List[Obstacle]):
        mid = Polygon([[0, -5], [0, 5], [60, 15], [60, -15]])
        left = Polygon([[0, -15], [0, -5], [60, -15], [50, -30]])
        right = Polygon([[0, 15], [0, 5], [60, 15], [50, 30]])
        # Span trapezoids
        angleInRad = (self.angle) * np.pi / 180
        dirVec = np.array([np.cos(angleInRad), np.sin(angleInRad)])
        posRot = np.add(dirVec * self.width / 2, self.get_position())

        zones = [mid, left, right]
        translated_zones = []

        for z in zones:
            z = affinity.translate(z, posRot[0], posRot[1])
            z = affinity.rotate(z, self.angle, posRot)
            translated_zones.append(z)

        for obj in obstacles:
            polygon_rotated = obj.polygon

            if translated_zones[0].intersects(polygon_rotated):
                return 0  # 1695  # Center
            if translated_zones[1].intersects(polygon_rotated):
                return 2  # 3215  # Left
            if translated_zones[2].intersects(polygon_rotated):
                return 1  # 2086  # Right
        return 3  # 4000  # Clear

    @property
    def direction_vector(self):
        return np.array([np.cos(self.angle), np.sin(self.angle)])

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

    def intersects(self, rayOrigin, rayDirection, p1, p2):
        rayOrigin = np.array(rayOrigin, dtype=np.float)
        rayDirection = np.divide(rayDirection, np.linalg.norm(rayDirection))
        point1 = np.array(p1, dtype=np.float)
        point2 = np.array(p2, dtype=np.float)
        v1 = rayOrigin - point1
        v2 = point2 - point1
        v3 = np.array([-rayDirection[1], rayDirection[0]])
        t1 = np.cross(v2, v1) / np.dot(v2, v3)
        t2 = np.dot(v1, v3) / np.dot(v2, v3)
        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return np.array([rayOrigin + t1 * rayDirection])
        return np.array([np.inf, np.inf])

    def reset(self, box: Box):
        if box.shape != (2,):
            raise ValueError("Can sample a point from a plain 2D box only")
        self.position = box.sample()
        self.angle = random.randint(0, 360)
