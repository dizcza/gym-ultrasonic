import math
import random
import time
import warnings
from typing import List, Tuple

import numpy as np
from gym.spaces import Box
from shapely import affinity
from shapely import speedups
from shapely.geometry import Polygon, LineString

if not speedups.enabled:
    warnings.warn("Install Cython to enable shapely speedups")


class Obstacle:
    """
    A static rectangular obstacle.
    """

    def __init__(self, position, width, height, angle=0):
        """
        Parameters
        ----------
        position: List[float]
            Obstacle center position in world coordinates, (x, y).
        width: float
            Obstacle width, mm
        height: float
            Obstacle height, mm
        angle: float
            Obstacle rotation angle in degrees.
            Positive angles are counter-clockwise and negative are clockwise rotations.
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

    def set_position(self, position):
        """
        Sets the object global position.

        Parameters
        ----------
        position: List[float]
            `Obstacle` center position in world coordinates, (x, y).
        """
        self.position = np.array(position, dtype=np.float32)

    def __str__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self):
        return f"{self.width} x {self.height}"


class ServoStub(Obstacle):
    """
    Servo stub that cannot rotate.
    """

    def __init__(self, width, height, angle_range=None):
        """
        Parameters
        ----------
        width: float
            Servo width, mm
        height: float
            Servo height, mm
        angle_range: Tuple[float]
            Min and max rotation angles, degrees.
            Has no effect for `ServoStub`.
            Default is `None`.
        """
        super().__init__(position=[0, 0], width=width, height=height, angle=0)
        self.angle_range = angle_range

    def rotate(self, angle_turn=None):
        """
        Does nothing.

        Parameters
        ----------
        angle_turn: float
            Angle to rotate the servo. Ignored.
        """
        pass

    def reset(self):
        """
        Resets the servo.
        """
        pass

    def extra_repr(self):
        return f"angle_range={self.angle_range}"


class Servo(ServoStub):
    """
    Servo that holds sonar on board.
    Servo has only one function - rotate the sonar.
    """

    def __init__(self, width, height, angle_range=(-90, 90), angular_vel=30):
        """
        Parameters
        ----------
        width: float
            Servo width, mm
        height: float
            Servo height, mm
        angle_range: Tuple[float]
            Min and max rotation angles, degrees
        angular_vel: float or str
            Rotation degrees per second.
        """
        super().__init__(width=width, height=height, angle_range=angle_range)
        self.angular_vel = angular_vel
        self.tick = None
        self.ccw = 1

    def rotate(self, angle_turn=None):
        """
        Rotates the servo.

        Parameters
        ----------
        angle_turn: float
            Angle to rotate the servo.
            If set to `None`, it's calculated by the time spent, multiplied by the angular velocity.
        """
        if self.tick is None:
            self.tick = time.time()
        tick = time.time()
        if angle_turn is None:
            angle_turn = (tick - self.tick) * self.angular_vel * self.ccw
        angle = self.angle + angle_turn
        self.tick = tick
        min_angle, max_angle = self.angle_range
        if angle > max_angle:
            angle = max_angle
            # start turning clockwise
            self.ccw = -1
        elif angle < min_angle:
            angle = min_angle
            # start turning counter-clockwise
            self.ccw = 1
        self.angle = angle

    def reset(self):
        """
        Resets the servo.
        """
        self.tick = None
        self.ccw = 1
        self.angle = 0

    def extra_repr(self):
        return f"{super().extra_repr()}, angular_vel={self.angular_vel}"


class Robot(Obstacle):
    """
    Robot with a mounted servo and ultrasonic range sensor.
    """

    def __init__(self, width, height, sensor_max_dist=2000, servo=None):
        """
        Initializes the robot with the given parameters at `[0, 0]` position and `0` rotation angle.

        Parameters
        ----------
        width: float
            Robot width, mm
        height: float
            Robot height, mm
        sensor_max_dist: float
            Ultrasonic sonar max range, mm.
            Default is 2000 mm.
        """
        super().__init__(position=[0., 0.], width=width, height=height, angle=0)
        self.sensor_max_dist = sensor_max_dist
        if servo is None:
            servo = ServoStub(width=0.3 * self.height, height=0.5 * self.width)
        self.servo = servo

    @property
    def servo_shift(self):
        """
        Returns
        -------
        float
            Shift of the servo along robot's main axis, mm
        """
        return 0.3 * self.width

    def move_forward(self, move_step):
        """
        Parameters
        ----------
        move_step: float
            Move step, mm.
        """
        vec = np.multiply(self.direction_vector, move_step)
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
            An obstacle(s) to check for a collision.

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
    def servo_position(self):
        """
        Returns
        -------
        np.ndarray
            World coordinates of Servo with embedded Ultrasonic sensor, (x, y)
        """
        return self.position + self.direction_vector * self.servo_shift

    def ray_cast(self, obstacles):
        """
        Casts a ray along servo sensor direction and checks if there an intersection with `obstacles`.
        Simulates Ultrasonic range sonar, mounted on top of the servo, in a real robot.

        Parameters
        ----------
        obstacles: List[Obstacle]
            List of obstacles in the scene.

        Returns
        -------
        min_dist: float
            Min dist to an obstacle.
            If no obstacle is found at the ray intersection, `max_dist` is returned.
        intersection_xy: list or np.ndarray
            X and Y of the intersection point with an obstacle.
        """
        if self.collision(obstacles):
            return 0., self.position
        sensor_pos = self.servo_position
        angle_target = math.radians(self.angle + self.servo.angle)
        target_direction = np.array([np.cos(angle_target), np.sin(angle_target)])
        ray_cast = LineString([sensor_pos, sensor_pos + target_direction * self.sensor_max_dist])
        min_dist = self.sensor_max_dist
        intersection_xy = [-self.sensor_max_dist, -self.sensor_max_dist]  # hide from a drawing screen

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

    def reset(self, box):
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
        self.servo.reset()

    def extra_repr(self):
        return f"{super().extra_repr()}, {self.servo}, sensor_max_dist={self.sensor_max_dist}"
