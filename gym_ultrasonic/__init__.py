from gym.envs.registration import register

register(
    id='Ultrasonic-v0',
    entry_point='gym_ultrasonic.envs:UltrasonicEnv',
    max_episode_steps=1000,
)
