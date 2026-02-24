import os
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from ..utils import line_chart, scatter_plot
from .utils import compute_averaged_convergence


class Visualizer:
    def __init__(self, xlim=None):
        self.xlim = xlim

    def plot_convergences(
        self, filepath, convergence_data, title=None, hide_legend=False
    ):
        if title is None:
            title = (
                "Convergence Comparison"
                if len(convergence_data) > 1
                else f'{convergence_data[0]["label"]} Convergence'
            )

        line_chart(
            convergence_data,
            title=title,
            xlabel="Evaluation Function Calls",
            ylabel="Best Fitness",
            withLegend=len(convergence_data) > 1 and not hide_legend,
            logscale=True,
            xlim=self.xlim,
            filepath=filepath,
        )

    def draw_averaged_convergence(self, filepath, *experiments):
        def extract_averaged_convergence(name, filepaths):
            evals, avg_fitness = compute_averaged_convergence(filepaths)
            return {"data": (evals.tolist(), avg_fitness.tolist()), "label": name}

        convergence_data = [
            extract_averaged_convergence(name, filepaths)
            for name, filepaths in experiments
        ]

        title = (
            "Convergence Comparison"
            if len(convergence_data) > 1
            else f'{convergence_data[0]["label"]} Convergence'
        )

        line_chart(
            convergence_data,
            title=title,
            xlabel="Evaluation Function Calls",
            ylabel="Best Fitness",
            withLegend=len(convergence_data) > 1,
            logscale=True,
            xlim=self.xlim,
            filepath=filepath,
        )

    def draw_all_seed_convergence(
        self,
        filepath,
        experiment_name,
        seed_dirs,
        title=None,
        title_fontsize=None,
        label_fontsize=None,
        legend_fontsize=None,
        legend_title_fontsize=None,
        params_text=None,
        params_fontsize=11,
    ):
        convergence_data = []
        dfs = []

        for seed_dir in seed_dirs:
            csv_path = f"{seed_dir}/out.csv"
            try:
                df = pd.read_csv(csv_path)
                convergence_data.append(
                    {
                        "data": (
                            df["Num of Evals"].tolist(),
                            df["Fitness Score"].tolist(),
                        ),
                        "label": os.path.basename(seed_dir),
                        "linestyle": ":",
                    }
                )
                dfs.append(df)
            except FileNotFoundError:
                print(f"⚠️ Missing: {csv_path}, skipping this seed.")
                continue

        if not convergence_data:
            print(f"No valid out.csv files found in {seed_dirs}")
            return

        min_eval = max(df["Num of Evals"].min() for df in dfs)
        max_eval = min(df["Num of Evals"].max() for df in dfs)

        common_evals = np.linspace(min_eval, max_eval, 500)

        interpolated_fitness = []
        for df in dfs:
            f = interp1d(
                df["Num of Evals"],
                df["Fitness Score"],
                kind="linear",
                fill_value="extrapolate",
            )
            interpolated_fitness.append(f(common_evals))

        average_fitness = np.mean(interpolated_fitness, axis=0)

        convergence_data.append(
            {
                "data": (common_evals.tolist(), average_fitness.tolist()),
                "label": "Average",
                "linestyle": "-",
            }
        )

        if title is None:
            title = f"{experiment_name} – All Seeds Convergence"

        plot_kwargs = {
            "title": title,
            "xlabel": "Evaluation Function Calls",
            "ylabel": "Fitness Score",
            "withLegend": False,
            "logscale": True,
            "xlim": self.xlim,
            "filepath": filepath,
        }

        if title_fontsize is not None:
            plot_kwargs["title_fontsize"] = title_fontsize
        if label_fontsize is not None:
            plot_kwargs["label_fontsize"] = label_fontsize
        if legend_fontsize is not None:
            plot_kwargs["legend_fontsize"] = legend_fontsize
        if legend_title_fontsize is not None:
            plot_kwargs["legend_title_fontsize"] = legend_title_fontsize

        if params_text is not None:
            plot_kwargs["params_text"] = params_text
            plot_kwargs["params_fontsize"] = params_fontsize
            plot_kwargs["params_position"] = "top_right"

        line_chart(convergence_data, **plot_kwargs)

    @staticmethod
    def draw_pareto(
        filepath,
        *experiments,
        connect_points=False,
        true_front=None,
        test_function=None,
        evals=None,
    ):
        pareto_data = [
            {
                "data": pd.read_csv(f"{filepath}/pareto-front.csv").iloc[:, -2:].values,
                "label": name,
            }
            for name, filepath in experiments
        ]

        if test_function and evals:
            title = f"{test_function} - QIEA ({evals} Evaluations)"
        else:
            title = (
                "Pareto Fronts Comparison"
                if len(pareto_data) > 1
                else f"{pareto_data[0]['label']} Pareto Front"
            )

        scatter_plot(
            pareto_data,
            title=title,
            xlabel="Objective 1",
            ylabel="Objective 2",
            withLegend=len(pareto_data) > 1,
            filepath=filepath,
            connect_points=False,
            true_front=true_front,
        )

    def set_xlim(self, xlim):
        self.xlim = xlim

    def draw_intermediate_pareto_fronts(
        self, experiment_name, seed_dirs, output_dir, n_intervals=4, true_front=None
    ):
        all_files = []
        for seed_dir in seed_dirs:
            files = [
                f for f in os.listdir(seed_dir) if f.startswith("pareto-front_gen-")
            ]
            all_files.extend([(seed_dir, f) for f in files])

        eval_counts = []
        for _, fname in all_files:
            match = re.search(r"evals-(\d+)", fname)
            if match:
                eval_counts.append(int(match.group(1)))

        if not eval_counts:
            return

        max_evals = max(eval_counts)
        interval = max_evals // n_intervals
        checkpoints = [interval * (i + 1) for i in range(n_intervals)]

        for checkpoint in checkpoints:
            intermediate_dir = os.path.join(
                output_dir, f"intermediate_{checkpoint//interval}-evals_{checkpoint}"
            )
            os.makedirs(intermediate_dir, exist_ok=True)
            pareto_data = []
            for seed_dir in seed_dirs:
                seed_name = os.path.basename(seed_dir)
                seed_num = seed_name.split("-")[1]
                # Find closest file to checkpoint
                seed_files = [
                    (f, int(re.search(r"evals-(\d+)", f).group(1)))
                    for f in os.listdir(seed_dir)
                    if f.startswith("pareto-front_gen-")
                    and re.search(r"evals-(\d+)", f)
                ]

                closest = min(seed_files, key=lambda x: abs(x[1] - checkpoint))
                df = pd.read_csv(os.path.join(seed_dir, closest[0]))
                pareto_data.append(
                    {"data": df.iloc[:, -2:].values, "label": f"Seed {seed_num}"}
                )

            scatter_plot(
                pareto_data,
                title=f"{experiment_name} - QIEA ({checkpoint} Evaluations)",
                xlabel="Objective 1",
                ylabel="Objective 2",
                withLegend=True,
                filepath=os.path.join(
                    intermediate_dir, f"all_seeds_eval_{checkpoint}.png"
                ),
                connect_points=False,
                true_front=true_front,
            )

            # Individual seed plots
            for i, seed_dir in enumerate(seed_dirs):
                seed_name = os.path.basename(seed_dir)
                scatter_plot(
                    [pareto_data[i]],
                    title=f"{experiment_name} - QIEA ({checkpoint} Evaluations)",
                    xlabel="Objective 1",
                    ylabel="Objective 2",
                    withLegend=False,
                    filepath=os.path.join(
                        intermediate_dir, f"{seed_name}_eval_{checkpoint}.png"
                    ),
                    connect_points=False,
                    true_front=true_front,
                )
