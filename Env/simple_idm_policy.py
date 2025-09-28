import numpy as np

class ConstantVelocityPolicy:
    def __init__(self, target_speed=50):
        self.step_num = 0

    def act(self):
        self.step_num += 1
        if self.step_num % 30 < 15:
            throttle = 1.0
        else:
            throttle = 1.0

        steering = 0.1

        # return [steering, throttle]

        return [0.0,0.05]
