from gym.envs.registration import register

# No servo
register(
    id='Ultrasonic-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicEnv',
    max_episode_steps=1000,
)

# Servo with a fixed angular velocity
register(
    id='UltrasonicServo-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    max_episode_steps=1000,
)


# With a servo. Rotation angle is predicted by the actor
register(
    id='UltrasonicServo-v1',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    kwargs={"servo_angular_vel": 'learn'},
    max_episode_steps=1000,
)
