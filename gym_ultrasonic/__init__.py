from gym.envs.registration import register

register(
    id='UltrasonicServo-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicServoEnv',
    timestep_limit=1000,
)
