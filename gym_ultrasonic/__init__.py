from gym.envs.registration import register

from .envs.constants import SONAR_DIRECTION_ANGLES

# Servo is fixed. One sonar, straight ahead.
register(
    id='Ultrasonic-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicEnv',
    kwargs={"sonar_direction_angles": (0,)},
    max_episode_steps=1000,
)

# Servo is fixed. Multiple sonars.
register(
    id='Ultrasonic-v1',
    entry_point='gym_ultrasonic.envs:UltrasonicEnv',
    kwargs={"sonar_direction_angles": SONAR_DIRECTION_ANGLES},
    max_episode_steps=1000,
)


# Servo with a fixed angular velocity
register(
    id='UltrasonicServo-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"sonar_direction_angles": (0,)},
    max_episode_steps=1000,
)


# Servo with learnable rotation angle (predicted by the actor)
register(
    id='UltrasonicServo-v1',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"sonar_direction_angles": (0,), "servo_angular_vel": 'learn'},
    max_episode_steps=1000,
)


# Servo with a fixed angular velocity. Servo angle is not included in the observation space.
register(
    id='UltrasonicServo-v2',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"sonar_direction_angles": (0,), "observe_servo_angle": False},
    max_episode_steps=1000,
)
