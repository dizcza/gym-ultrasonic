import math

ROBOT_WIDTH = 115  # mm
ROBOT_HEIGHT = 140  # mm
SENSOR_DIST_MAX = 2000  # max dist in mm ultrasonic sensor can capture
SERVO_ANGLE_MAX = math.radians(30)
WHEEL_VELOCITY_MAX = 145  # mm/s
SERVO_ANGULAR_VELOCITY = 0.5  # rad/s
SIMULATION_TIME_STEP = 0.5  # sec
SONAR_DIRECTION_ANGLES = (-SERVO_ANGLE_MAX, 0, SERVO_ANGLE_MAX)

SCREEN_SIZE = 3000  # mm
SCREEN_SCALE_DOWN = 5
N_OBSTACLES = 4

# servo_angular_vel == 'learn' mode
SONAR_TURN_ANGLE_MAX_LEARN_MODE = math.radians(20)
