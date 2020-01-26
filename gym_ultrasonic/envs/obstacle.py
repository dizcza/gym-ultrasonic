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

from .constants import SENSOR_DIST_MAX, SERVO_ANGULAR_VELOCITY, SERVO_ANGLE_MAX

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
            Obstacle rotation angle in radians.
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
        coords = self.get_polygon_parallel_coords() + self.position
        polygon_parallel = Polygon(coords)
        return affinity.rotate(polygon_parallel, self.angle, use_radians=True)

    def get_polygon_parallel_coords(self):
        """
        Returns
        -------
        np.ndarray
            Clock-wise vertices of the bounding box polygon, parallel to the world axes.
        """
        # zero angle is aligned with Ox axis
        # that's why width is substituted by height
        l, r, t, b = -self.height / 2, self.height / 2, self.width / 2, -self.width / 2
        return np.array([(l, b), (l, t), (r, t), (r, b)], dtype=np.float32)

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


class Servo(Obstacle):
    """
    Servo that holds sonar(s) on board.
    Servo has only one function - rotate the sonar(s).
    """

    def __init__(self, width, height, sonar_direction_angles=(0,), angle_range=(-SERVO_ANGLE_MAX, SERVO_ANGLE_MAX),
                 angular_vel=0):
        """
        Parameters
        ----------
        width: float
            Servo width, mm
        height: float
            Servo height, mm
        sonar_direction_angles: Tuple[float]
            List of sonar directions angles, radians.
        angle_range: Tuple[float]
            Min and max rotation angles, radians
        angular_vel: float or str
            Rotation radians per second.
        """
        super().__init__(position=[0, 0], width=width, height=height, angle=0)
        self.sonar_direction_angles = sonar_direction_angles
        self.angle_range = angle_range
        self.angular_vel = angular_vel
        self.ccw = 1

    @property
    def sonar_angles_world(self):
        return np.add(self.angle, self.sonar_direction_angles)

    def rotate(self, sim_time=None, angle_turn=None):
        """
        Rotates the servo.

        Parameters
        ----------
        sim_time: float
            Simulation time step, sec.
        angle_turn: float
            Angle to rotate the servo.
            If set to `None`, it's calculated by the time spent, multiplied by the angular velocity.
        """
        if sim_time is None and angle_turn is None:
            raise ValueError("Either sim_time or angle_turn must be specified.")
        if angle_turn is None:
            angle_turn = sim_time * self.angular_vel * self.ccw
        if angle_turn == 0:
            # do nothing
            return
        angle = self.angle + angle_turn
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
        self.ccw = 1
        self.angle = 0

    def extra_repr(self):
        to_degrees = lambda rads: '(' + ', '.join(f"{math.degrees(rad):.1f}" for rad in rads) + ')'
        return f"sonar_direction_angles={to_degrees(self.sonar_direction_angles)} degrees, " \
               f"angle_range={to_degrees(self.angle_range)} degrees, " \
               f"angular_vel={math.degrees(self.angular_vel):.1f} degrees/s"


class Robot(Obstacle):
    """
    Robot with a mounted servo and ultrasonic range sensor.
    """

    def __init__(self, width, height, sonar_direction_angles=(0,), sensor_max_dist=SENSOR_DIST_MAX):
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
            Default is `SENSOR_MAX_DIST`.
        """
        super().__init__(position=[0., 0.], width=width, height=height, angle=0)
        self.sensor_max_dist = sensor_max_dist

        # servo is 90 degrees rotated w.r.t. robot
        # that's why width is substituted by height
        self.servo = Servo(width=0.5 * self.height, height=0.25 * self.width,
                           sonar_direction_angles=sonar_direction_angles,
                           angular_vel=0)

    @property
    def servo_shift(self):
        """
        Returns
        -------
        float
            Shift of the servo along robot's main axis, mm
        """
        return 0.3 * self.height

    @property
    def n_sonars(self):
        """
        Returns
        -------
        int
            Num. of sonars on board.
        """
        return len(self.servo.sonar_direction_angles)

    def move_forward(self, move_step):
        """
        Parameters
        ----------
        move_step: float
            Move step, mm.
        """
        vec = np.multiply(self.direction_vector, move_step)
        self.position += vec

    def diffdrive(self, vel_left, vel_right, sim_time):
        """
        Differential drive. Adapted from
        http://ais.informatik.uni-freiburg.de/teaching/ss17/robotics/exercises/solutions/03/sheet03sol.pdf

        Parameters
        ----------
        vel_left: float
            Left wheel tangential speed, mm / sec.
        vel_right: float
            Right wheel tangential speed, mm / sec.
        sim_time: float
            Simulation time step, sec.

        Returns
        -------
        move_step: float
            Performed move step in mm along robot's main axis.
        angle_rotate: float
            Performed rotation in radians.
        trajectory: LineString
            Trajectory of this step.
        """

        def rotation_matrix(theta):
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array([[c, -s],
                             [s, c]])

        old_pos = np.copy(self.position)

        if vel_left == vel_right:
            # straight line
            move_step = vel_left * sim_time
            angle_rotate = 0
            self.move_forward(move_step)
            trajectory = LineString([old_pos, self.position])
        else:  # circular motion
            # Calculate the radius of rotation
            r_icc = self.width / 2.0 * ((vel_left + vel_right) / (vel_right - vel_left))

            # compute instantaneous center of curvature
            x, y = self.position
            x_icc = x - r_icc * np.sin(self.angle)
            y_icc = y + r_icc * np.cos(self.angle)

            # compute the angular velocity
            omega = (vel_right - vel_left) / self.width

            # compute the angle change
            angle_rotate = omega * sim_time

            # forward kinematics for differential drive
            pos_icc = np.array([x - x_icc, y - y_icc])  # robot position in ICC coordinate system
            rotate = lambda angle: rotation_matrix(angle).dot(pos_icc) + [x_icc, y_icc]
            self.position = rotate(angle_rotate)

            move_step = r_icc * angle_rotate  # arc length
            self.angle += angle_rotate

            # create curvature trajectory for rendering
            dtheta_intermediate = np.linspace(0, angle_rotate, num=5, endpoint=True)
            trajectory = LineString(map(rotate, dtheta_intermediate))

        # dilate line
        trajectory = trajectory.buffer(distance=self.width / 2., cap_style=3, resolution=2)
        return move_step, angle_rotate, trajectory

    def turn(self, angle_step):
        """
        Parameters
        ----------
        angle_step: float
            CCW angle turn in radians.
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
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def servo_position(self):
        """
        Returns
        -------
        np.ndarray
            World coordinates of Servo with embedded Ultrasonic sensor, (x, y)
        """
        return self.position + self.direction_vector * self.servo_shift

    def _ray_cast_single_sonar(self, obstacles, sonar_angle):
        sensor_pos = self.servo_position
        angle_target = self.angle + sonar_angle
        target_direction = np.array([np.cos(angle_target), np.sin(angle_target)])
        ray_cast = LineString([sensor_pos, sensor_pos + target_direction * self.sensor_max_dist])

        min_dist = self.sensor_max_dist
        # hide the intersection circle from a drawing screen
        intersection_xy = np.array([-self.sensor_max_dist, -self.sensor_max_dist])

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
        min_dist: List[float]
            Min dist to an obstacle.
            If no obstacle is found at the ray intersection, `max_dist` is returned.
        intersection_xy: List[np.ndarray]
            X and Y of the intersection point with an obstacle.
        """
        if self.collision(obstacles):
            min_dist_array = [0.] * self.n_sonars
            intersection_xy_array = np.tile(self.position, reps=(self.n_sonars, 1)).tolist()
        else:
            min_dist_array = []
            intersection_xy_array = []
            for sonar_angle in self.servo.sonar_angles_world:
                min_dist, intersection_xy = self._ray_cast_single_sonar(obstacles, sonar_angle=sonar_angle)
                min_dist_array.append(min_dist)
                intersection_xy_array.append(intersection_xy)
        return min_dist_array, intersection_xy_array

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
        self.angle = random.uniform(0, 2 * math.pi)
        self.servo.reset()

    def extra_repr(self):
        return f"{super().extra_repr()}, {self.servo}, sensor_max_dist={self.sensor_max_dist} mm"
