import math
import random


class QuBitSim:
    def __init__(
        self,
        angle_of_rotation,
        theta=0,
        bound=math.pi / 2,
        mutation_probability_rotation=0.0,
        mutation_probability_measurement=0.0,
    ):
        self.angle_of_rotation = angle_of_rotation
        self.bounds = (-bound, bound)
        self.theta = theta
        self.mutation_probability_rotation = mutation_probability_rotation
        self.mutation_probability_measurement = mutation_probability_measurement

    def rotate(self, direction=1):
        new_theta = self.theta + direction * self.angle_of_rotation

        if new_theta < self.bounds[0]:
            new_theta = self.bounds[0]

        elif new_theta > self.bounds[1]:
            new_theta = self.bounds[1]

        if random.random() < self.mutation_probability_rotation:
            new_theta *= -1

        self.theta = new_theta

    def measure(self):
        probability_of_zero = math.sin(math.pi / 4 - self.theta / 2) ** 2
        random_number = random.random()
        bit = 0 if random_number < probability_of_zero else 1

        if random.random() < self.mutation_probability_measurement:
            bit = 1 - bit

        return bit
