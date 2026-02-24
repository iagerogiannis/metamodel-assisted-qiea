import numpy as np
import random
import math
import csv
from pathlib import Path

from ...db import ParquetDB, InMemoryDB
from ...utils import read_config, flatten
from ...optimization.metamodels import MOMetamodel
from ..utils import (
    linear_distribution,
    linear_weighted_choice,
    mo_sort,
    probabilistic_tournament_selection,
)
from .population import Population
from .individual import QEAIndividual
from .stagnation_control import StagnationControl
from .io_ops import IOOPS, bcolors


class QEASolver:
    def __init__(self, problem_params, optimization_params, io_params):
        self.design_variables = problem_params["design_variables"]
        self.multi_objective = problem_params["multi_objective"]
        self.evolution_strategy = optimization_params["evolution_strategy"]
        self.max_num_of_evals = self.evolution_strategy["termination_criteria"][
            "max_num_of_evals"
        ]
        self.target_fitness = self.evolution_strategy["termination_criteria"].get(
            "target_fitness", None
        )
        self.metamodel_config = optimization_params.get("metamodel", {}).get(
            "config", None
        )

        self.io_ops = IOOPS(io_params, self.design_variables, self.multi_objective)
        self.db = (
            ParquetDB(io_params["output_dir"], problem_params["multi_objective"])
            if io_params["db"] == "parquet"
            else InMemoryDB()
        )
        self.db.clear()
        self.metamodel = None
        self.min_num_of_samples_for_metamodel = None
        self.exact_evaluations_per_generation_percentage = 1.0
        self.num_training_patterns = 0
        self.generation_index = 0
        self.num_of_evals = 0
        self.population = None

        if self.metamodel_config:
            metamodel_config = read_config(self.metamodel_config)
            self.min_num_of_samples_for_metamodel = metamodel_config["min_num_samples"]
            self.num_training_patterns = metamodel_config["num_training_patterns"]
            self.exact_evaluations_per_generation_percentage = metamodel_config[
                "exact_evaluations_per_generation_percentage"
            ]
            self.metamodel = MOMetamodel(
                metamodel_config,
                problem_params["num_of_objectives"] if self.multi_objective else 1,
            )

        def surrogate_model(*args, id=None, **kwargs):
            def find_nearest_neighbors(args):
                data = self.db.get_all()

                inputs = np.stack(data["input"].to_numpy())
                current = np.array(args[0]) if len(args) == 1 else np.array(args)

                distances = np.linalg.norm(inputs - current, axis=1)

                nearest_indices = np.argsort(distances)[: self.num_training_patterns]
                nearest_neighbors = [data.iloc[i] for i in nearest_indices]

                return nearest_neighbors

            retrieved_value = self.db.get_by_id(id)
            if retrieved_value is not None:
                return retrieved_value, True

            nearest_neighbors = find_nearest_neighbors(args)
            X = np.array([row["input"] for row in nearest_neighbors])
            y = np.array([row["output"] for row in nearest_neighbors])
            estimation = self.metamodel.predict(np.array(list(args)), X, y)

            if estimation:
                return estimation, False

            return None, False

        def wrapped_fitness_function(*args, id=None, exact=False, **kwargs):
            retrieved_value = self.db.get_by_id(id)

            if retrieved_value is not None:
                return retrieved_value, True

            if (
                self.metamodel
                and not exact
                and self.num_of_evals > self.min_num_of_samples_for_metamodel
            ):
                estimation, is_exact = surrogate_model(*args, id=id)

                if estimation is not None:
                    return estimation, is_exact

            self.num_of_evals += 1
            exact_value = problem_params["fitness_function"](*args, **kwargs)
            self.db.put(id, self.generation_index, list(args), exact_value)

            return exact_value, True

        self.surrogate_model = surrogate_model
        self.fitness_function = wrapped_fitness_function

        # Populations
        self.parents_population_size = self.evolution_strategy["populations"][
            "parents_population_size"
        ]
        self.offspring_population_size = self.evolution_strategy["populations"][
            "offspring_population_size"
        ]
        self.elites_population_size = self.evolution_strategy["populations"][
            "elites_population_size"
        ]

        # Migration
        self.global_migration_period = self.evolution_strategy["migration"][
            "global_migration_period"
        ]
        self.local_migration_period = self.evolution_strategy["migration"][
            "local_migration_period"
        ]
        self.neighbourhood_size = self.evolution_strategy["migration"][
            "neighbourhood_size"
        ]

        # Quantum Rotation
        self.angle_of_rotation = self.evolution_strategy["quantum_rotation"].get(
            "angle_of_rotation", 1.0
        )

        self.measurements_per_individual = self.evolution_strategy[
            "quantum_rotation"
        ].get("measurements_per_individual", 1)

        self.qubit_bound = self.evolution_strategy["quantum_rotation"].get(
            "qubit_bound", math.pi / 2
        )

        self.mutation_probability_rotation = self.evolution_strategy[
            "quantum_rotation"
        ].get("mutation_probability_rotation", 0.0)

        self.mutation_probability_measurement = self.evolution_strategy[
            "quantum_rotation"
        ].get("mutation_probability_measurement", 0.0)

        self.rotation_table = self.evolution_strategy["quantum_rotation"].get(
            "rotation_table", None
        )

        # Parent selection
        self.elitism_rate = self.evolution_strategy["parents_selection"].get(
            "elitism_rate", 0.0
        )
        self.tournament_size = self.evolution_strategy["parents_selection"].get(
            "tournament_size", 3
        )
        self.include_elites_chance = self.evolution_strategy["parents_selection"].get(
            "include_elites_chance", 0.2
        )

        # Crossover
        self.crossover_rate = self.evolution_strategy["crossover"].get(
            "crossover_rate", 0.2
        )
        self.crossover_period = self.evolution_strategy["crossover"].get(
            "crossover_period", 10
        )
        self.crossover_schema = self.evolution_strategy["crossover"].get(
            "crossover_schema", "single_point/var"
        )

        self.log_chromosomes = io_params.get("log_chromosomes", False)
        if self.log_chromosomes:
            self.write_every = io_params.get("write_every", 10)
        self.all_time_best = None

        self.stagnation_control = StagnationControl(
            self.evolution_strategy["stagnation_control"]
        )

    def initialize_population(self):
        self.parents_population = Population(
            [], self.multi_objective, self.parents_population_size
        )
        self.elites_population = Population(
            [], self.multi_objective, self.elites_population_size
        )
        self.population = self._init_pop(self._get_offspring_population_size())

        if not self.multi_objective:
            self.update_all_time_best()

    def get_latest_generation(self, force_output=False):
        self.elites_selection()
        latest_generation = {
            "index": self.generation_index,
            "num_of_evals": self.num_of_evals,
            "opt": self.elites_population.get_optimal(),
        }
        self.io_ops.output_result(latest_generation, force_output)

        if self.log_chromosomes and self.generation_index % self.write_every == 0:
            self._log_chromosomes()

        return latest_generation

    def update_all_time_best(self):
        current_best = min(
            [i.best_measurement for i in self.population.pop],
            key=lambda x: x["fitness_score"],
        )

        if (
            self.all_time_best is None
            or current_best["fitness_score"] < self.all_time_best["fitness_score"]
        ):
            self.all_time_best = current_best

    def perform_migrations(self, generation_index):
        def global_migration():
            for individual in self.population.pop:
                individual.migrate(self.all_time_best)

        def local_migration():
            neighboorhoods = [
                self.population.pop[i : i + self.neighbourhood_size]
                for i in range(0, len(self.population.pop), self.neighbourhood_size)
            ]
            for i, neighbourhood in enumerate(neighboorhoods):
                best_neighbour = min(
                    [i.best_measurement for i in neighbourhood],
                    key=lambda x: x["fitness_score"],
                )
                for j, _ in enumerate(neighbourhood):
                    self.population.pop[i * self.neighbourhood_size + j].migrate(
                        best_neighbour
                    )

        # if generation_index < 1:
        #     return

        if (
            self.global_migration_period
            and generation_index % self.global_migration_period == 0
        ):
            global_migration()

        elif (
            self.neighbourhood_size
            and self.local_migration_period
            and generation_index % self.local_migration_period == 0
        ):
            local_migration()

    def apply_crossover(self, generation_index, force=False, tournament_size=3):
        if generation_index % self.crossover_period != 0 and not force:
            return

        offspring = []

        # for _ in range((self.population.max_population_size + 1) // 2):
        #     parent_pair = [
        #         probabilistic_tournament_selection(
        #             self.parents_population.pop,
        #             tournament_size,
        #             key=lambda x: x.fitness_score,
        #         )
        #         for _ in range(2)
        #     ]

        #     if random.random() < self.crossover_rate:
        #         # newborn = random.choice(parent_pair[0].crossover(parent_pair[1]))
        #         # newborn = min(
        #         #     parent_pair[0].crossover(parent_pair[1]),
        #         #     key=lambda ind: ind.best_measurement["fitness_score"],
        #         # )
        #         newborns = parent_pair[0].crossover(parent_pair[1])
        #         offspring.extend(newborns)

        #     else:
        #         offspring.extend([parent_pair[random.randint(0, 1)].clone()])

        # self.population.colonize(offspring)
        # self.population.sort()
        # self.population.eliminate_overpopulation()

        for _ in range(self.population.max_population_size):
            parent_pair = [
                probabilistic_tournament_selection(
                    (
                        self.population.pop
                        if self.multi_objective
                        else self.parents_population.pop
                    ),
                    tournament_size,
                    key=lambda x: x.fitness_score,
                )
                for _ in range(2)
            ]

            if random.random() < self.crossover_rate:
                offspring.extend(
                    [
                        random.choice(
                            parent_pair[0].crossover(
                                parent_pair[1],
                                self.multi_objective,
                                self.crossover_schema,
                            )
                        )
                    ]
                )

            else:
                offspring.extend([parent_pair[random.randint(0, 1)].clone()])

        self.population.colonize(offspring, append=self.multi_objective)

    def elitism(self, renew=False):
        num_of_elites = (
            min(
                max(int(self.elitism_rate * self.parents_population_size), 1),
                len(self.elites_population.pop),
            )
            if random.random() < self.include_elites_chance and not renew
            else 0
        )

        return random.sample(self.elites_population.pop, num_of_elites)

    def parent_selection(self, tournament_size=3, renew=False, method="tournament"):
        elites = self.elitism(renew)

        newborn_pop = (
            self._init_pop(self.parents_population_size // 2).pop if renew else []
        )

        parents_pop = []
        if method == "tournament":
            parents_pop = [
                (
                    probabilistic_tournament_selection(
                        self.parents_population.pop + self.population.pop,
                        tournament_size,
                        key=lambda x: x.fitness_score,
                    ).clone()
                )
                for _ in range(
                    self.parents_population_size - len(elites) - len(newborn_pop)
                )
            ]
        else:
            parents_pop = [
                i.clone()
                for i in linear_weighted_choice(
                    self.parents_population.pop + self.population.pop,
                    self.parents_population_size - len(elites) - len(newborn_pop),
                    key=lambda x: x.fitness_score,
                )
            ]

        self.parents_population.colonize(elites + parents_pop + newborn_pop)

    def elites_selection(self):
        self.elites_population.colonize(
            self.elites_population.pop + self.population.pop
        ).sort().eliminate_overpopulation()

    def evolve_single_objective(self, generation_index, metamodel_assisted):
        stagnation_detected, shaking_gen = self.stagnation_control.detect_stagnation(
            generation_index, self.num_of_evals
        )

        self.resize_offspring_population()
        self.parent_selection(stagnation_detected, method="linear_weighted")
        self.apply_crossover(generation_index, shaking_gen)

        cycle_best = min(
            [
                i.evolve_single_objective(
                    self.measurements_per_individual if not metamodel_assisted else 20,
                    metamodel_assisted,
                )
                for i in self.population.pop
            ]
        )

        self.update_all_time_best()
        self.perform_migrations(generation_index)
        self.stagnation_control.update_current_cycle_best(self.num_of_evals, cycle_best)

    def evolve_multi_objective_non_assisted(self, generation_index):
        self.apply_crossover(generation_index)

        self.population.colonize(
            flatten(
                [
                    i.evolve_multi_objective_non_assisted(
                        self.measurements_per_individual
                    )
                    for i in self.population.pop
                ]
            )
        )
        self.population.eliminate_duplicates()
        self.population.sort()
        self.population.eliminate_overpopulation()

    def evolve_multi_objective_metamodel_assisted(self):
        self.apply_crossover()

        evolution = [
            i.evolve_multi_objective_metamodel_assisted(
                self.measurements_per_individual
            )
            for i in self.population.pop
        ]
        evolved_pop = flatten([ind[0] for ind in evolution])
        num_exact_evals = sum([ind[1] for ind in evolution])

        inexact_evolved_pop = [
            ind for ind in evolved_pop if not ind.latest_measurement["exact"]
        ]

        num_individuals = len(evolved_pop)
        num_desired_exact_evals = int(
            num_individuals * self.exact_evaluations_per_generation_percentage
        )
        num_pending_exact_evals = max(
            num_desired_exact_evals - num_exact_evals, len(inexact_evolved_pop)
        )

        if num_pending_exact_evals > 0:
            latest_fitness = lambda ind: ind.latest_measurement["fitness_score"]
            sorted_inexact_evolved_pop = mo_sort(inexact_evolved_pop, latest_fitness)
            evolved_pop += flatten(
                [
                    ind.upgrade_to_exact_mo()
                    for ind in sorted_inexact_evolved_pop[:num_pending_exact_evals]
                ]
            )

        self.population.colonize(evolved_pop)
        self.population.eliminate_duplicates()
        self.population.sort()
        self.population.eliminate_overpopulation()

    def evolve(self, generation_index):
        metamodel_assisted = (
            self.metamodel is not None
            and self.num_of_evals > self.min_num_of_samples_for_metamodel
        )

        if not self.multi_objective:
            self.evolve_single_objective(generation_index, metamodel_assisted)

        if self.multi_objective and not metamodel_assisted:
            self.evolve_multi_objective_non_assisted(generation_index)

        if self.multi_objective and metamodel_assisted:
            self.evolve_multi_objective_metamodel_assisted()

    def solve(self):
        self.initialize_population()
        latest_generation = self.get_latest_generation()

        while self.num_of_evals < self.max_num_of_evals:
            self.generation_index += 1
            self.evolve(self.generation_index)
            latest_generation = self.get_latest_generation()

            if (
                not self.multi_objective
                and latest_generation["opt"]["fitness"] < self.target_fitness
            ):
                print(
                    f"{bcolors.WARNING}Num of evals: {bcolors.ENDC}"
                    f"{bcolors.OKCYAN}{self.num_of_evals}{bcolors.ENDC}"
                )

                self.io_ops.output_result(latest_generation, force_output=True)
                self.num_of_evals = self.max_num_of_evals
                latest_generation = self.get_latest_generation()

        self.io_ops.output_result(latest_generation, force_output=True)
        self.io_ops.print_summary(latest_generation)
        self.io_ops.close_out_file()

    def reset(self, seed):
        self.generation_index = 0
        self.num_of_evals = 0
        self.population = None
        self.db.clear()
        self.io_ops.reset(seed)
        self.stagnation_control.reset()
        self.all_time_best = None

    def resize_offspring_population(self):
        if self.offspring_population_size["type"] != "linear":
            return

        self.population.max_population_size = self._get_offspring_population_size()

    def _init_pop(self, size):
        return Population(
            [
                QEAIndividual(
                    self.design_variables,
                    self.fitness_function,
                    self.surrogate_model,
                    self.multi_objective,
                    self.angle_of_rotation,
                    self.mutation_probability_rotation,
                    self.mutation_probability_measurement,
                    self.rotation_table,
                    self.qubit_bound,
                    None,
                    None,
                    None,
                )
                for _ in range(size)
            ],
            self.multi_objective,
        )

    def _log_chromosomes(self):
        chromosome_log_dir = Path(self.io_ops.output_directory) / "chromosome_logs"
        chromosome_log_dir.mkdir(parents=True, exist_ok=True)

        for idx, individual in enumerate(self.population.pop):
            filepath = chromosome_log_dir / f"individual_{idx}.csv"

            thetas = [qubit.theta for qubit in individual.chromosome]

            file_exists = filepath.exists()
            with open(filepath, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        ["generation", "num_of_evals"]
                        + [f"qubit_{i}" for i in range(len(thetas))]
                    )
                writer.writerow([self.generation_index, self.num_of_evals] + thetas)

    def _get_offspring_population_size(self):
        if self.offspring_population_size["type"] not in ["constant", "linear"]:
            raise ValueError(
                f"Unknown offspring_population_size type: {self.offspring_population_size['type']}. "
                "Supported types are 'constant' and 'linear'."
            )

        if self.offspring_population_size["type"] == "constant":
            return self.offspring_population_size["value"]

        if self.offspring_population_size["type"] == "linear":
            return linear_distribution(
                self.num_of_evals - self.stagnation_control.reset_at_evals,
                self.offspring_population_size["x0"],
                self.offspring_population_size["y0"],
                self.offspring_population_size["x1"],
                self.offspring_population_size["y1"],
                min_value=self.offspring_population_size["min_val"],
                max_value=self.offspring_population_size["max_val"],
            )
