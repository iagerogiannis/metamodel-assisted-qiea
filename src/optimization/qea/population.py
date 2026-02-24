from ..utils import mo_sort, get_pareto_front
from .individual import QEAIndividual


class Population:
    def __init__(
        self,
        pop: list[QEAIndividual],
        multi_objective: bool,
        max_population_size: int = None,
    ):
        self.pop = pop
        self.multi_objective = multi_objective
        self.max_population_size = (
            max_population_size if max_population_size else len(pop)
        )

    def sort(self):
        if self.multi_objective:
            self.pop = mo_sort(self.pop)

        else:
            self.pop.sort(key=lambda x: x.fitness_score)

        return self

    def select(self, num_of_individuals, exact=False):
        if exact:
            return [i for i in self.pop if i.exact][:num_of_individuals]

        return self.pop[:num_of_individuals]

    def eliminate_duplicates(self):
        unique_pop = {}
        for individual in self.pop:
            sig = individual.signature()
            if sig not in unique_pop:
                unique_pop[sig] = individual

        self.pop = list(unique_pop.values())

        return self

    def eliminate_overpopulation(self):
        self.pop = self.select(self.max_population_size)
        return self

    def colonize(self, pop, append=False):
        if append:
            self.pop.extend(pop)

        else:
            self.pop = pop

        return self

    def get_optimal(self):
        pop = [i for i in self.pop if i.exact is True]

        if self.multi_objective:
            pareto_front = get_pareto_front(pop)
            return [
                {
                    "solution": individual.decoded_solution,
                    "fitness": individual.fitness_score,
                }
                for individual in pareto_front
            ]

        min_index = min(range(len(pop)), key=lambda i: pop[i].fitness_score)

        return {
            "solution": pop[min_index].decoded_solution,
            "fitness": pop[min_index].fitness_score,
        }

    def set_max_population_size(self, size: int):
        self.max_population_size = size
