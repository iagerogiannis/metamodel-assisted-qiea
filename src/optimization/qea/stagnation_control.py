import numpy as np


class StagnationControl:
    def __init__(self, config):
        # stagnated_evals_threshold: 200 for metamodel assisted
        self.stagnated_evals_threshold = config.get("stagnated_evals_threshold", 600)
        self.num_shaking_gens = config.get("num_shaking_gens", 5)

        self.current_cycle_best_fitness = np.inf
        self.current_cycle_num_stagnated_evals = 0
        self.last_improvement_at_evals = 0
        self.reset_at_evals = 0
        self.reset_at_gen = 0

    def detect_stagnation(self, generation_index, num_evals):
        if self.current_cycle_num_stagnated_evals > self.stagnated_evals_threshold:
            self.reset_at_evals = num_evals
            self.current_cycle_best_fitness = np.inf
            self.reset_at_gen = generation_index

        after_initialization = num_evals > 100

        stagnation_detected = (
            after_initialization and generation_index - self.reset_at_gen < 1
        )
        shaking_gen = (
            after_initialization
            and generation_index - self.reset_at_gen < self.num_shaking_gens
        )

        return (
            stagnation_detected,
            shaking_gen,
        )

    def update_current_cycle_best(self, num_evals, cycle_best_fitness):
        if cycle_best_fitness < self.current_cycle_best_fitness:
            self.current_cycle_best_fitness = cycle_best_fitness
            self.last_improvement_at_evals = num_evals

        self.current_cycle_num_stagnated_evals = (
            num_evals - self.last_improvement_at_evals
        )

    def reset(self):
        self.last_improvement_at_evals = 0
        self.current_cycle_num_stagnated_evals = 0
        self.reset_at_evals = 0
        self.reset_at_gen = 0
        self.current_cycle_best_fitness = np.inf
