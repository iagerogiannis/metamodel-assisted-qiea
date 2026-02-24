import os
import re
import random
from .qea import QEASolver
from .test_functions import get_test_function, get_true_pareto_front
from .executable_fitness_function import ExecutableFitnessFunction
from .visualizer import Visualizer


class Optimizer:
    def __init__(
        self,
        experiment_params,
        problem_params,
        optimization_params,
        io_params,
        auto_visualize=False,
    ):
        self.fitness_function_name = problem_params.get("fitness_function")

        if problem_params.get("use_executable", False):
            problem_params["fitness_function"] = ExecutableFitnessFunction(
                problem_params["fitness_function"],
                timeout=problem_params.get("timeout", 30),
            )
        else:
            problem_params["fitness_function"] = get_test_function(
                problem_params["fitness_function"]
            )

        self.experiment_params = experiment_params
        self.problem_params = problem_params
        self.optimization_params = optimization_params
        self.io_params = io_params
        self.auto_visualize = auto_visualize

        self.solver = QEASolver(problem_params, optimization_params, io_params)

    def run(self):
        print()
        print(f"Running {self.experiment_params['name']}...")

        output_base_dir = self.io_params["output_dir"]
        seed_filepaths = []

        is_multi_objective = self.problem_params.get("multi_objective", False)

        for _seed in range(self.experiment_params["shots"]):
            seed = _seed + self.experiment_params["seed_value"]
            random.seed(seed)

            seed_dir = os.path.join(output_base_dir, f"seed-{seed}")
            out_csv_path = os.path.join(
                seed_dir, "out.csv" if not is_multi_objective else "pareto-front.csv"
            )

            if os.path.exists(out_csv_path):
                print(f"Seed {seed}... (skipping, out.csv already exists)")
            else:
                print(f"Seed {seed}...")
                self.solver.reset(seed)
                self.solver.solve()

            seed_filepaths.append(seed_dir)

        if self.auto_visualize:
            self._auto_visualize_results()

        return seed_filepaths

    def _auto_visualize_results(self):
        visualizer = Visualizer()
        output_dir = self.io_params["output_dir"]
        output_plots_dir = self.io_params.get("output_plots_dir", output_dir)
        os.makedirs(output_plots_dir, exist_ok=True)

        experiment_name = self.experiment_params["name"]
        is_multi_objective = self.problem_params.get("multi_objective", False)

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]", "", experiment_name.replace(" ", "_")
        ).lower()

        seed_dirs = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("seed-") and os.path.isdir(os.path.join(output_dir, d))
        ]

        if not seed_dirs:
            print(f"No seed directories found in {output_dir}, skipping visualization.")
            return

        if not is_multi_objective:
            avg_path = os.path.join(output_plots_dir, f"{safe_name}-convergence.png")
            all_seeds_path = os.path.join(
                output_plots_dir, f"{safe_name}-all-seeds-convergence.png"
            )

            print(f"Drawing averaged convergence plot: {avg_path}")
            visualizer.draw_averaged_convergence(avg_path, (experiment_name, seed_dirs))

            print(f"Drawing all-seed convergence plot: {all_seeds_path}")
            path = os.path.join(
                output_plots_dir,
                f"{safe_name}-all-seeds-convergence.png",
            )
            visualizer.draw_all_seed_convergence(path, experiment_name, seed_dirs)
            return

        true_front = (
            get_true_pareto_front(self.fitness_function_name)
            if self.fitness_function_name
            else None
        )

        max_evals = self.optimization_params.get("max_function_evaluations", None)

        pareto_outputs = []
        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            seed_num = seed_name.split("-")[1]
            pareto_path = os.path.join(
                output_plots_dir, f"{safe_name}-{seed_name}-pareto.png"
            )
            print(f"Drawing Pareto front for {seed_name}: {pareto_path}")
            visualizer.draw_pareto(
                pareto_path,
                (f"Seed {seed_num}", seed_dir),
                connect_points=False,
                true_front=true_front,
                test_function=self.fitness_function_name,
                evals=max_evals,
            )
            pareto_outputs.append((f"Seed {seed_num}", seed_dir))

        combined_path = os.path.join(
            output_plots_dir, f"{safe_name}-all-seeds-pareto.png"
        )
        print(f"Drawing combined Pareto front: {combined_path}")
        visualizer.draw_pareto(
            combined_path,
            *pareto_outputs,
            connect_points=False,
            true_front=true_front,
            test_function=self.fitness_function_name,
            evals=max_evals,
        )

        visualizer.draw_intermediate_pareto_fronts(
            experiment_name,
            seed_dirs,
            output_plots_dir,
            n_intervals=3,
            true_front=true_front,
        )
