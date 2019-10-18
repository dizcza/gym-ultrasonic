# OpenAI Gym Ultrasonic robot sensor environment

[![Build Status](https://travis-ci.org/dizcza/gym-ultrasonic.svg?branch=master)](https://travis-ci.org/dizcza/gym-ultrasonic)
[![Coverage Status](https://coveralls.io/repos/github/dizcza/gym-ultrasonic/badge.svg?branch=master)](https://coveralls.io/github/dizcza/gym-ultrasonic?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gym-ultrasonic/badge/?version=latest)](https://gym-ultrasonic.readthedocs.io/en/latest/?badge=latest)

![](docs/images/one-sonar.png)

### Environment

Adapted from https://github.com/lelmac/robotsim (originally, `AutonomousRobot-v1`).

`Ultrasonic-v0` - Ultrasonic sonar sensor (without a moving servo), mounted on top of a robot (small red rectangle), heads forward.

`UltrasonicServo-v0` - Ultrasonic sonar sensor with a servo that has fixed angular velocity.

`UltrasonicServo-v1` - Ultrasonic sonar sensor with a servo that learns how to rotate the sensor.


## Setup

Requires Python 3.6+

```bash
pip install -r requirements.txt
python ddpg.py
```
