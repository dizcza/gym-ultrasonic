from gym.envs.registration import register

# Without a servo (zero angular velocity).
register(
    id='UltrasonicServo-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"servo_angular_vel": 0},  # turn servo off
    max_episode_steps=1000,
)

# With a servo.
register(
    id='UltrasonicServo-v1',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    max_episode_steps=1000,
)


# With a servo. Rotation angle is predicted by the actor
register(
    id='UltrasonicServo-v2',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"servo_angular_vel": 'learn'},
    max_episode_steps=1000,
)
