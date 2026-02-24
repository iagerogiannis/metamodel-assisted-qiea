import math
from ..utils import flatten, dominates
from ..chromosome_decoder import ChromosomeDecoder
from .qubuit_simulator import QuBitSim
import random


class QEAIndividual:
    def __init__(
        self,
        design_variables,
        fitness_function,
        surrogate_model,
        multi_objective,
        angle_of_rotation,
        mutation_probability_rotation=0.0,
        mutation_probability_measurement=0.0,
        rotation_table=None,
        qubit_bound=math.pi / 2,
        chromosome=None,
        latest_measurement=None,
        best_measurement=None,
    ):
        self.measurements_count = 0
        self.fruitless_measurements_count = 0
        self.surrogate_model = surrogate_model
        self.angle_of_rotation = angle_of_rotation
        self.mutation_probability_rotation = mutation_probability_rotation
        self.mutation_probability_measurement = mutation_probability_measurement
        # Default rotation table matches original behavior
        # [x=0,b=0,better], [x=0,b=1,better], [x=1,b=0,better], [x=1,b=1,better],
        # [x=0,b=0,!better], [x=0,b=1,!better], [x=1,b=0,!better], [x=1,b=1,!better]
        self.rotation_table = (
            rotation_table
            if rotation_table is not None
            else [0, 0, 0, 0, "randint", 1, -1, "randint"]
        )
        self.qubit_bound = qubit_bound
        self.best_measurement = dict(best_measurement) if best_measurement else None
        self.latest_measurement = (
            dict(latest_measurement) if latest_measurement else None
        )
        self.recent_measurements = []
        self.most_promising = None
        self.debt = 0

        self.design_variables = design_variables
        self.fitness_function = fitness_function
        self.multi_objective = multi_objective
        self.decoder = ChromosomeDecoder(design_variables)
        self.chromosome = chromosome if chromosome else self.initialize_chromosome()
        self.fitness_score = None
        self.decoded_solution = None
        self.exact = None

        if best_measurement:
            self.exact = best_measurement["exact"]

        if not latest_measurement:
            self.latest_measurement = self.measure(exact=True)

        if not best_measurement:
            self.update_best_measurement()

        self.fitness_score = self.best_measurement["fitness_score"]
        self.decoded_solution = self.best_measurement["decoded_value"]

        self.eligible_for_parency = True

    def initialize_chromosome(self):
        num_of_qubits = sum(variable["bits"] for variable in self.design_variables)
        return [
            QuBitSim(
                self.angle_of_rotation,
                bound=self.qubit_bound,
                mutation_probability_rotation=self.mutation_probability_rotation,
                mutation_probability_measurement=self.mutation_probability_measurement,
            )
            for _ in range(num_of_qubits)
        ]

    def measure(self, exact=False, with_surrogate=False):
        if exact:
            self.measurements_count += 1
            self.fruitless_measurements_count += 1

        encoded_value = [quBit.measure() for quBit in self.chromosome]
        return self.evaluate_fitness(encoded_value, exact, with_surrogate)

    def evaluate_fitness(self, encoded_value, exact=False, with_surrogate=False):
        eval_function = (
            self.fitness_function
            if exact or not with_surrogate
            else self.surrogate_model
        )
        decoded_value = self.decode(encoded_value)
        fitness_score, exactly_evaluated = eval_function(
            *decoded_value, id=self._sign(encoded_value), exact=exact
        )
        return {
            "encoded_value": encoded_value,
            "decoded_value": decoded_value,
            "fitness_score": fitness_score,
            "exact": exactly_evaluated,
        }

    def update_chromosome(self, latest_measurement_is_better):
        def get_direction_for_qu_bit(x_i, b_i):
            nonlocal latest_measurement_is_better
            # Calculate index in rotation table
            # Index = (0 if better else 4) + (x_i * 2) + b_i
            base_index = 0 if latest_measurement_is_better else 4
            index = base_index + (x_i * 2) + b_i

            # Get value from rotation table
            value = self.rotation_table[index]

            # Process special string values
            if value == "randint":
                return random.choice([-1, 1])
            elif value == "randfloat":
                return random.uniform(-1, 1)
            else:
                # Numeric value
                return value

        for i, quBit in enumerate(self.chromosome):
            direction = get_direction_for_qu_bit(
                self.latest_measurement["encoded_value"][i],
                self.best_measurement["encoded_value"][i],
            )
            quBit.rotate(direction)

    def update_best_measurement(self):
        if not self.latest_measurement["exact"]:
            raise ValueError(
                "Cannot update best measurement with inexact latest measurement"
            )

        self.best_measurement = dict(self.latest_measurement)
        self.fitness_score = self.best_measurement["fitness_score"]
        self.decoded_solution = self.best_measurement["decoded_value"]
        self.exact = self.best_measurement["exact"]

    def characterize_latest_measurement(self):
        if not self.multi_objective:
            return (
                "DOMINANT"
                if self.latest_measurement["fitness_score"]
                < self.best_measurement["fitness_score"]
                else "DOMINATED"
            )

        root_dominates = dominates(
            self.best_measurement["fitness_score"],
            self.latest_measurement["fitness_score"],
        )
        evolution_dominates = dominates(
            self.latest_measurement["fitness_score"],
            self.best_measurement["fitness_score"],
        )

        if root_dominates:
            return "DOMINATED"

        if evolution_dominates:
            return "DOMINANT"

        return "NON_DOMINATED_NON_DOMINANT"

    def fission(self, latest_measurement=None):
        chromosome_duplicate = [
            QuBitSim(
                self.angle_of_rotation,
                qubit.theta,
                bound=self.qubit_bound,
                mutation_probability_rotation=self.mutation_probability_rotation,
                mutation_probability_measurement=self.mutation_probability_measurement,
            )
            for qubit in self.chromosome
        ]
        return QEAIndividual(
            self.design_variables,
            self.fitness_function,
            self.surrogate_model,
            self.multi_objective,
            self.angle_of_rotation,
            self.mutation_probability_rotation,
            self.mutation_probability_measurement,
            self.rotation_table,
            self.qubit_bound,
            chromosome_duplicate,
            (
                dict(latest_measurement)
                if latest_measurement
                else dict(self.latest_measurement)
            ),
            dict(self.best_measurement),
        )

    def clone(self):
        return QEAIndividual(
            self.design_variables,
            self.fitness_function,
            self.surrogate_model,
            self.multi_objective,
            self.angle_of_rotation,
            self.mutation_probability_rotation,
            self.mutation_probability_measurement,
            self.rotation_table,
            self.qubit_bound,
            [
                QuBitSim(
                    self.angle_of_rotation,
                    qubit.theta,
                    bound=self.qubit_bound,
                    mutation_probability_rotation=self.mutation_probability_rotation,
                    mutation_probability_measurement=self.mutation_probability_measurement,
                )
                for qubit in self.chromosome
            ],
            dict(self.latest_measurement),
            dict(self.best_measurement),
        )

    def migrate(self, best_measurement):
        self.best_measurement = dict(best_measurement)
        self.fitness_score = self.best_measurement["fitness_score"]
        self.decoded_solution = self.best_measurement["decoded_value"]
        self.exact = self.best_measurement["exact"]

    def exactly_evaluate_latest_measurement(self):
        if not self.latest_measurement["exact"]:
            self.latest_measurement = self.evaluate_fitness(
                self.latest_measurement["encoded_value"], exact=True
            )

        return self.characterize_latest_measurement()

    def decode(self, measurement):
        return self.decoder.decode(measurement)

    def signature(self):
        return self._sign([str(round(qubit.theta, 1)) for qubit in self.chromosome])
        # return self._sign(self.best_measurement["encoded_value"])
        # return self._sign(
        #     self.best_measurement["encoded_value"]
        #     + [str(round(qubit.theta, 1)) for qubit in self.chromosome]
        # )

    def _sign(self, encoded_value):
        return "".join(map(str, encoded_value))

    def crossover(
        self, other: "QEAIndividual", new_measurement=False, schema="single_point/var"
    ):
        if schema not in ["single_point/var", "two_point/var"]:
            raise ValueError(f"Invalid schema: {schema}")

        if schema == "single_point/var":
            crossover_point = sum(
                [
                    self.design_variables[i]["bits"]
                    for i in range(random.randint(1, len(self.design_variables) - 1))
                ]
            )

            offspring1_chromosome = [
                QuBitSim(
                    self.angle_of_rotation,
                    qubit.theta,
                    bound=self.qubit_bound,
                    mutation_probability_rotation=self.mutation_probability_rotation,
                    mutation_probability_measurement=self.mutation_probability_measurement,
                )
                for qubit in (
                    self.chromosome[:crossover_point]
                    + other.chromosome[crossover_point:]
                )
            ]

            offspring2_chromosome = [
                QuBitSim(
                    self.angle_of_rotation,
                    qubit.theta,
                    bound=self.qubit_bound,
                    mutation_probability_rotation=self.mutation_probability_rotation,
                    mutation_probability_measurement=self.mutation_probability_measurement,
                )
                for qubit in (
                    other.chromosome[:crossover_point]
                    + self.chromosome[crossover_point:]
                )
            ]

        else:  # two_point/var
            point1 = random.randint(1, len(self.design_variables) - 1)
            point2 = random.randint(1, len(self.design_variables) - 1)
            if point1 > point2:
                point1, point2 = point2, point1

            cp1 = sum([self.design_variables[i]["bits"] for i in range(point1)])
            cp2 = sum([self.design_variables[i]["bits"] for i in range(point2)])

            offspring1_chromosome = [
                QuBitSim(
                    self.angle_of_rotation,
                    qubit.theta,
                    bound=self.qubit_bound,
                    mutation_probability_rotation=self.mutation_probability_rotation,
                    mutation_probability_measurement=self.mutation_probability_measurement,
                )
                for qubit in (
                    self.chromosome[:cp1]
                    + other.chromosome[cp1:cp2]
                    + self.chromosome[cp2:]
                )
            ]

            offspring2_chromosome = [
                QuBitSim(
                    self.angle_of_rotation,
                    qubit.theta,
                    bound=self.qubit_bound,
                    mutation_probability_rotation=self.mutation_probability_rotation,
                    mutation_probability_measurement=self.mutation_probability_measurement,
                )
                for qubit in (
                    other.chromosome[:cp1]
                    + self.chromosome[cp1:cp2]
                    + other.chromosome[cp2:]
                )
            ]

        measurement_1 = (
            dict((self if random.random() < 0.5 else other).latest_measurement)
            if not new_measurement
            else None
        )

        measurement_2 = (
            dict((self if random.random() < 0.5 else other).latest_measurement)
            if not new_measurement
            else None
        )

        return [
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
                offspring1_chromosome,
                measurement_1,
                measurement_1,
            ),
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
                offspring2_chromosome,
                measurement_2,
                measurement_2,
            ),
        ]

    def evolve_individual_mo(self):
        latest_measurement_characterization = self.characterize_latest_measurement()

        if latest_measurement_characterization != "DOMINATED":
            latest_measurement_characterization = (
                self.exactly_evaluate_latest_measurement()
            )

        if latest_measurement_characterization == "NON_DOMINATED_NON_DOMINANT":
            new_individual = self.fission()

            self.update_chromosome(False)

            new_individual.update_chromosome(True)
            new_individual.update_best_measurement()

            return [self, new_individual], sum(
                [
                    self.latest_measurement["exact"],
                    new_individual.latest_measurement["exact"],
                ]
            )

        latest_measurement_is_better = (
            self.characterize_latest_measurement() == "DOMINANT"
        )

        self.update_chromosome(latest_measurement_is_better)

        if latest_measurement_is_better:
            self.update_best_measurement()

        return [self], sum([self.latest_measurement["exact"]])

    def get_most_promising_inexact_measurement(self):
        """Returns the most promising inexact measurement"""
        inexact = [m for m in self.recent_measurements if not m["exact"]]
        return min(inexact, key=lambda m: m["fitness_score"]) if inexact else None

    def upgrade_to_exact_mo(self):
        self.exactly_evaluate_latest_measurement()
        return self.evolve_individual_mo()[0]

    def assign_latest_and_update(self, measurements):
        self.latest_measurement = min(measurements, key=lambda m: m["fitness_score"])

        latest_measurement_is_better = (
            self.characterize_latest_measurement() == "DOMINANT"
        )

        self.update_chromosome(latest_measurement_is_better)

        if latest_measurement_is_better:
            self.fruitless_measurements_count = 0
            self.update_best_measurement()

    def evolve_single_objective(self, num_of_measurements=1, metamodel_assisted=False):
        if not metamodel_assisted:
            measurements = [
                self.measure(exact=True) for _ in range(num_of_measurements)
            ]
            self.assign_latest_and_update(measurements)

        else:
            approximations_count = 0
            approximations = []
            while approximations_count < num_of_measurements:
                measurement = self.measure(with_surrogate=True)
                if (
                    measurement["fitness_score"] is not None
                    and not measurement["exact"]
                ):
                    approximations.append(measurement)
                    approximations_count += 1

            approximations.sort(key=lambda m: m["fitness_score"])

            self.assign_latest_and_update(
                [
                    self.evaluate_fitness(
                        approximations[0]["encoded_value"], exact=True
                    ),
                ]
            )

        return self.latest_measurement["fitness_score"]

    def evolve_multi_objective_non_assisted(self, num_of_measurements):
        measurements = [self.measure(exact=True) for _ in range(num_of_measurements)]
        return flatten(
            [self.fission(m).evolve_individual_mo()[0] for m in measurements]
        )

    def evolve_multi_objective_metamodel_assisted(self, num_of_measurements):
        measurements = [self.measure() for _ in range(num_of_measurements)]
        evolution = [self.fission(m).evolve_individual_mo() for m in measurements]
        evolved_pop = flatten([evo[0] for evo in evolution])
        num_exact_evals = sum([evo[1] for evo in evolution])
        return evolved_pop, num_exact_evals

    def reset(self, hard=False):
        if hard:
            for qubit in self.chromosome:
                qubit.theta = 0
        else:
            # TODO: Parameterize reset angle percentage
            reset_angle = 0.25 * self.angle_of_rotation
            for qubit in self.chromosome:
                if qubit.theta > 0:
                    qubit.theta = max(0, qubit.theta - reset_angle)
                elif qubit.theta < 0:
                    qubit.theta = min(0, qubit.theta + reset_angle)

    def set_eligible_for_parency(self, eligible: bool):
        self.eligible_for_parency = eligible

    def force_egalitarianism(self, num_of_measurements):
        self.reset()
        measurements = [self.measure(exact=True) for _ in range(num_of_measurements)]
        best = min(measurements, key=lambda m: m["fitness_score"])
        self.latest_measurement = best
        self.best_measurement = best
        self.measurements_count = 0
