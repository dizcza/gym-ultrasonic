# OpenAI Gym Ultrasonic robot sensor environment

[![Build Status](https://travis-ci.org/dizcza/gym-ultrasonic.svg?branch=master)](https://travis-ci.org/dizcza/gym-ultrasonic)
[![Coverage Status](https://coveralls.io/repos/github/dizcza/gym-ultrasonic/badge.svg?branch=master)](https://coveralls.io/github/dizcza/gym-ultrasonic?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gym-ultrasonic/badge/?version=latest)](https://gym-ultrasonic.readthedocs.io/en/latest/?badge=latest)

![](docs/images/one-sonar.png)

### Environment
`Ultrasonic-v0` (`AutonomousRobot-v1` in the [original repo](https://github.com/lelmac/robotsim)) - Ultrasonic sensor (without moving servo for now), mounted on top of a robot (small red circle), heads forward.

## Setup
```bash
pip install -r requirements.txt
python ddpg.py
```
