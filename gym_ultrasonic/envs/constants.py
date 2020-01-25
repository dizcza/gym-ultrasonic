import math

ROBOT_WIDTH = 115  # mm
ROBOT_HEIGHT = 140  # mm
SENSOR_DIST_MAX = 2000  # max dist in mm ultrasonic sensor can capture
SERVO_ANGLE_MAX = math.radians(30)
WHEEL_VELOCITY_MAX = 45  # mm/s
SERVO_ANGULAR_VELOCITY = 0.5  # rad/s
SIMULATION_TIME_STEP = 0.5  # sec

SCREEN_SIZE = 3000  # mm
SCREEN_SCALE_DOWN = 5

# hack recurrence: augment input with the last N-1 observations
OBSERVATIONS_MEMORY_SIZE = 1
