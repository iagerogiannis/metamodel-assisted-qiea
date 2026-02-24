import os
from ..optimization import Optimizer, Visualizer
from ..utils import read_config


def run_optimization_experiments(configs_dir="config_examples"):
    config_files = [
        os.path.join(configs_dir, f)
        for f in os.listdir(configs_dir)
        if f.endswith(".json")
    ]

    experiment_params = [read_config(f) for f in config_files]

    all_filepaths = {}

    for params in experiment_params:
        optimizer = Optimizer(
            experiment_params=params["experiment"],
            problem_params=params["problem"],
            optimization_params=params["optimization"],
            io_params=params["io_params"],
            auto_visualize=True,
        )

        result_paths = optimizer.run()
        all_filepaths[params["experiment"]["name"]] = result_paths

    return all_filepaths


def visualize_results(all_filepaths, output_dir):
    visualizer = Visualizer()

    so_experiments = []
    mo_experiments = []

    for exp_name, paths in all_filepaths.items():
        has_out_csv = any(os.path.exists(os.path.join(p, "out.csv")) for p in paths)
        has_pareto_csv = any(
            os.path.exists(os.path.join(p, "pareto-front.csv")) for p in paths
        )

        if has_out_csv and not has_pareto_csv:
            so_experiments.append((exp_name, paths))
        elif has_pareto_csv and not has_out_csv:
            mo_experiments.append((exp_name, paths[0]))

    os.makedirs(output_dir, exist_ok=True)

    if so_experiments:
        print("Generating single-objective comparison plot...")
        visualizer.draw_averaged_convergence(
            os.path.join(output_dir, "so-comparison.png"),
            *so_experiments,
        )

    if mo_experiments:
        print("Generating multi-objective Pareto front comparison plot...")
        visualizer.draw_pareto(
            os.path.join(output_dir, "pareto-front-comparison.png"),
            *mo_experiments,
            connect_points=False,
        )
