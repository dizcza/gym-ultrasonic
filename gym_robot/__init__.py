from gym.envs.registration import register

register(
    id='AutonomousRobot-v1',
    entry_point='gym_robot.envs:AutonomousRobotC',
    timestep_limit=1000,
)
