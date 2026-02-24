import argparse
import os
from src.experiment_runner import run_optimization_experiments, visualize_results


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--run-examples",
        action="store_true",
        help="Run the default examples for demo purposes",
    )

    group.add_argument("--config", type=str, help="Path to a single JSON config file")
    group.add_argument(
        "--config-dir",
        type=str,
        help="Path to a directory containing multiple JSON config files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/plots",
        help="Directory to save results and plots",
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization (only run experiments)",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Run only experiments whose name contains this substring",
    )

    args = parser.parse_args()

    if args.run_examples:
        print("Running default example experiments from 'config_examples'...\n")
        all_filepaths = run_optimization_experiments("config_examples")
        visualize_results(all_filepaths, "output/plots")
        return

    if args.config:
        all_filepaths = run_optimization_experiments(
            configs_dir=os.path.dirname(args.config)
        )
        all_filepaths = {
            k: v
            for k, v in all_filepaths.items()
            if args.config.split(os.sep)[-1].startswith(k)
        }

    else:
        all_filepaths = run_optimization_experiments(configs_dir=args.config_dir)
        if args.filter:
            all_filepaths = {
                k: v
                for k, v in all_filepaths.items()
                if args.filter.lower() in k.lower()
            }

    if not args.no_visualize:
        print(f"Generating visualizations in {args.output_dir}...")
        visualize_results(all_filepaths, args.output_dir)


if __name__ == "__main__":
    main()
