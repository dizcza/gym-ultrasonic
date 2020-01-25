import math
import random
from typing import List

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from shapely.geometry import LineString, MultiLineString

from .constants import WHEEL_VELOCITY_MAX, SERVO_ANGLE_MAX, SENSOR_DIST_MAX, ROBOT_HEIGHT, ROBOT_WIDTH, \
    SCREEN_SCALE_DOWN, SCREEN_SIZE, SERVO_ANGULAR_VELOCITY, SIMULATION_TIME_STEP, \
    N_OBSTACLES
from .obstacle import Robot, Obstacle


class UltrasonicEnv(gym.Env):
    """
    A differential robot with one Ultrasonic sonar sensor, trying to avoid obstacles. Never stops (no target).
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    # action space box is slightly larger because of the additive noise
    action_space = spaces.Box(low=-WHEEL_VELOCITY_MAX, high=WHEEL_VELOCITY_MAX, shape=(2,))

    def __init__(self, n_obstacles=N_OBSTACLES, time_step=SIMULATION_TIME_STEP,
                 sonar_direction_angles=(0,)):
        """
        Parameters
        ----------
        n_obstacles: int
            Num. of obstacles on the scene.
        time_step: float
            Simulation time step, used in :func:`Robot.diffdrive` and :func:`Servo.rotate`.
        sonar_direction_angles: Tuple[float]
            List of sonar direction angles.
        """
        super().__init__()
        self.time_step = time_step
        self.width = self.height = SCREEN_SIZE

        # robot's position will be reset later on
        self.robot = Robot(width=ROBOT_WIDTH, height=ROBOT_HEIGHT, sonar_direction_angles=sonar_direction_angles)

        # dist to obstacles
        self.observation_space = spaces.Box(low=0., high=SENSOR_DIST_MAX, shape=(self.robot.n_sonars,))

        wall_size = 10
        indent = wall_size + max(self.robot.width, self.robot.height)
        self.allowed_screen_space = spaces.Box(low=indent, high=self.width - indent, shape=(2,))

        walls = [
            Obstacle([0, self.height / 2], self.height, wall_size),  # left wall
            Obstacle([self.width, self.height / 2], self.height, wall_size),  # right wall
            Obstacle([self.width / 2, self.height], wall_size, self.width),  # top wall
            Obstacle([self.width / 2, 0], wall_size, self.width)  # bottom wall
        ]
        self.obstacles = []

        sample_obstacle_size = lambda: random.randint(int(0.75 * self.robot.width), self.robot.width * 5)

        for random_obstacle_id in range(n_obstacles):
            obst = Obstacle(position=random.sample(range(self.width), k=2),
                            width=sample_obstacle_size(),
                            height=sample_obstacle_size(),
                            angle=random.randint(0, 360))
            self.obstacles.append(obst)
        self.obstacles.extend(walls)

        self.state = self.init_state

        # rendering
        self.viewer = None
        self.ray_collision_transforms = [rendering.Transform() for sonar_id in range(self.robot.n_sonars)]
        self.robot_transform = rendering.Transform()
        self.servo_transform = rendering.Transform()
        self._current_trajectory = None

        self.reset()

    def step(self, action):
        """
        Make a single step:
            1) move forward by `action[0]` mm;
            2) rotate by `action[1]` radians;
            3) rotate servo by `action[2]` radians (`UltrasonicServoEnv-v1` only).

        Parameters
        ----------
        action: List[float]
            Env action to perform.

        Returns
        -------
        observation: List[float]
            A list that contains one value - min dist to an obstacle on its way.
        reward: float
            A reward, obtained by taking this `action`.
        done: bool
            Whether the episode has ended.
        info: dict
            An empty dict. Unused.
        """
        vel_left, vel_right = action[:2]
        servo_turn = action[2] if len(action) == 3 else None
        move_step, angle_rotate, trajectory = self.robot.diffdrive(vel_left, vel_right, sim_time=self.time_step)
        self._current_trajectory = trajectory
        self.robot.servo.rotate(sim_time=self.time_step, angle_turn=servo_turn)
        reward, done = self.reward(trajectory, move_step, angle_rotate, servo_turn)
        self.state = self.update_state()
        info = dict(move_step=move_step, angle_rotate=angle_rotate)
        return self.state, reward, done, info

    def update_state(self):
        """
        Returns
        -------
        state: List[float]
            Min dist to obstacles.
        """
        min_dist_array, _ = self.robot.ray_cast(self.obstacles)
        return min_dist_array

    def reward(self, trajectory, move_step, angle_rotate, servo_turn):
        """
        Computes the reward.

        Parameters
        ----------
        trajectory: LineString
            Trajectory of this step.
        move_step: float
            Move robot with `move_step` mm along its main axis.
        angle_rotate: float
            Turn robot by `angle_turn` radians.
        servo_turn: float
            Turn servo by `servo_turn` radians, if not None.

        Returns
        -------
        reward: float
            A reward, obtained by applying this step.
        done: bool
            Whether the robot collided with any of obstacles.
        """
        if self.robot.collision(self.obstacles):
            return -1000, True
        for obstacle in self.obstacles:
            if obstacle.polygon.intersects(trajectory):
                return -1000, True
        reward = -1 + move_step - 3 * abs(angle_rotate)
        # print(move_step, angle_rotate, reward)
        if servo_turn is not None:
            reward -= abs(servo_turn)
        return reward, False

    @property
    def init_state(self):
        """
        Returns
        -------
        List[float]
            Initial env state (observation) which is the min dist to obstacles.
            It's not clear what the default "min dist to obstacles" is - 0, `sensor_max_dist` or a value in-between.
            But since we `update_state()` after each `reset()`, it should not matter.
        """
        state_size = np.prod(self.observation_space.shape)
        return np.zeros(state_size, dtype=float).tolist()

    def reset(self):
        """
        Resets the state and spawns a new robot position.
        """
        self.state = self.init_state
        self.robot.reset(box=self.allowed_screen_space)
        while self.robot.collision(self.obstacles):
            self.robot.reset(box=self.allowed_screen_space)
        self.state = self.update_state()
        return self.state

    def init_view(self):
        """
        Initializes the Viewer (for displaying purpose only).
        """
        self.viewer = rendering.Viewer(self.width // SCREEN_SCALE_DOWN, self.height // SCREEN_SCALE_DOWN)
        robot_view = rendering.FilledPolygon(self.robot.get_polygon_parallel_coords() / SCREEN_SCALE_DOWN)
        robot_view.add_attr(self.robot_transform)
        robot_view.set_color(r=0., g=0., b=0.9)

        # ultrasonic sensor collision circle
        for intersection_transform in self.ray_collision_transforms:
            circle_collision = rendering.make_circle(radius=7)
            circle_collision.add_attr(intersection_transform)
            circle_collision.set_color(1, 0, 0)
            self.viewer.add_geom(circle_collision)

        servo_view = rendering.FilledPolygon(self.robot.servo.get_polygon_parallel_coords() / SCREEN_SCALE_DOWN)
        servo_view.add_attr(self.servo_transform)
        servo_view.add_attr(rendering.Transform(translation=(self.robot.servo_shift / SCREEN_SCALE_DOWN, 0)))
        servo_view.add_attr(self.robot_transform)
        servo_view.set_color(1, 0, 0)

        for obstacle in self.obstacles:
            polygon_coords = np.array(obstacle.polygon.boundary.coords, dtype=np.float32) / SCREEN_SCALE_DOWN
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

        if self._current_trajectory is not None:
            trajectory_boundary = self._current_trajectory.boundary
            if isinstance(trajectory_boundary, MultiLineString):
                trajectory_linestrings = trajectory_boundary.geoms
            elif isinstance(trajectory_boundary, LineString):
                trajectory_linestrings = [trajectory_boundary]
            else:
                raise ValueError(f"Invalid trajectory boundary: {trajectory_boundary}")
            for linestring in trajectory_linestrings:
                trajectory_coords = np.array(linestring.coords, dtype=np.float32) / SCREEN_SCALE_DOWN
                trajectory_view = rendering.FilledPolygon(trajectory_coords)
                trajectory_view._color.vec4 = (0, 0, 0.9, 0.2)
                self.viewer.add_onetime(trajectory_view)

        _, intersection_xy_array = self.robot.ray_cast(self.obstacles)
        intersection_xy_array = np.divide(intersection_xy_array, SCREEN_SCALE_DOWN)
        for intersection_xy, intersection_transform in zip(intersection_xy_array, self.ray_collision_transforms):
            intersection_transform.set_translation(*intersection_xy)

        x_robot, y_robot = self.robot.position / SCREEN_SCALE_DOWN
        self.robot_transform.set_translation(x_robot, y_robot)
        self.robot_transform.set_rotation(self.robot.angle)
        self.servo_transform.set_rotation(self.robot.servo.angle)
        with_rgb = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=with_rgb)

    def __str__(self):
        return f"{super().__str__()}:\nobservation_space={self.observation_space.low, self.observation_space.high};" \
               f"\naction_space={self.action_space.low, self.action_space.high};" \
               f"\n{self.robot};" \
               f"\nnum. of obstacles: {len(self.obstacles) - 4}"  # 4 walls


class UltrasonicServoEnv(UltrasonicEnv):
    """
    A robot with one Ultrasonic sonar sensor and a servo that rotates the sonar.
    The task is the same: avoid obstacles. Never stops (no target).
    """

    def __init__(self, n_obstacles=N_OBSTACLES, time_step=SIMULATION_TIME_STEP,
                 sonar_direction_angles=(0,),
                 servo_angular_vel=SERVO_ANGULAR_VELOCITY):
        """
        Parameters
        ----------
        n_obstacles: int
            Num. of obstacles on the scene.
        time_step: float
            Simulation time step, used in :func:`Robot.diffdrive` and :func:`Servo.rotate`.
        sonar_direction_angles: Tuple[float]
            List of sonar direction angles.
        servo_angular_vel: float or str
            Servo angular velocity, radians per sec.
        """
        super().__init__(n_obstacles=n_obstacles, time_step=time_step, sonar_direction_angles=sonar_direction_angles)

        # dist to obstacles, servo_angle
        self.observation_space = spaces.Box(
            low=np.r_[self.observation_space.low, -SERVO_ANGLE_MAX],
            high=np.r_[self.observation_space.high, SERVO_ANGLE_MAX]
        )

        if servo_angular_vel == 'learn':
            # wheels left and right velocity (mm/s), servo turn (radians)
            upperbound = np.array([WHEEL_VELOCITY_MAX, WHEEL_VELOCITY_MAX, math.radians(20)])
            self.action_space = spaces.Box(low=-upperbound, high=upperbound)
        self.robot.servo.angular_vel = servo_angular_vel

    def update_state(self):
        """
        Returns
        -------
        state: List[float]
            Min dist to obstacles and servo rotation angle.
        """
        min_dist_array, _ = self.robot.ray_cast(self.obstacles)
        new_state = min_dist_array + [self.robot.servo.angle]
        return new_state
