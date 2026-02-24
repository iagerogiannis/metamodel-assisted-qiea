import pandas as pd
from os import path as os_path
from operator import xor
from ...utils import prepare_folder


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class IOOPS:
    def __init__(self, io_params, design_variables, multi_objective):
        self.print_every = io_params["print_every"]
        self.write_every = io_params["write_every"]
        self.output_dir_base = io_params["output_dir"]
        self.output_directory = self.output_dir_base

        self.multi_objective = multi_objective

        self.dv_names = [dv["name"] for dv in design_variables]
        self.dv_headers = [f"DV {i} ({name})" for i, name in enumerate(self.dv_names)]
        self.headers = (
            ["Gen Index", "Num of Evals"] + self.dv_headers + ["Fitness Score"]
        )

        self.out_file = None

    def print_formatted(self, data):
        print(
            "".join(
                [
                    f"{str(value):15s}" if i < 2 else f"{str(value):25s}"
                    for i, value in enumerate(data)
                ]
            )
        )

    def initialize_out_file(self):
        prepare_folder(self.output_directory)

        if self.multi_objective:
            return

        out_file_path = os_path.join(self.output_directory, "out.csv")
        out_file = open(out_file_path, "a")
        out_file.write(",".join(self.headers) + "\n")

        return out_file

    def print_headers(self):
        if self.multi_objective:
            return

        self.print_formatted(self.headers)

    def print_result(self, result):
        if self.multi_objective:
            # TODO: Finetune formatting
            return print("Num of evals: ", result["num_of_evals"])

        self.print_formatted(
            [result["index"]]
            + [result["num_of_evals"]]
            + result["opt"]["solution"]
            + [result["opt"]["fitness"]]
        )

    def write_result(self, result):
        if self.multi_objective:
            fname = os_path.join(
                self.output_directory,
                f'pareto-front_gen-{result["index"]}_evals-{result['num_of_evals']}.csv',
            )
            out_fname = os_path.join(self.output_directory, "pareto-front.csv")
            num_of_objectives = len(result["opt"][0]["fitness"])

            pareto_df = pd.DataFrame(
                [
                    list(solution["solution"]) + list(solution["fitness"])
                    for solution in result["opt"]
                ]
            )
            pareto_df.columns = self.dv_headers + [
                f"Objective {i}" for i in range(num_of_objectives)
            ]
            pareto_df.to_csv(fname, index=False)
            pareto_df.to_csv(out_fname, index=False)

        if not self.multi_objective:
            self.out_file.write(
                ",".join(
                    [str(result["index"]), str(result["num_of_evals"])]
                    + [str(value) for value in result["opt"]["solution"]]
                    + [str(result["opt"]["fitness"])]
                )
                + "\n"
            )

    def output_result(self, result, force_output=False):
        index = result["index"]

        if xor(force_output, index % self.print_every == 0):
            self.print_result(result)

        if xor(force_output, index % self.write_every == 0):
            self.write_result(result)

    def print_summary(self, result):
        print(30 * "=")
        print("-- Summary --")
        print(f'Number of generations: {result["index"] + 1}')
        print(f'Number of evaluations: {result["num_of_evals"]}')
        if not self.multi_objective:
            print(
                "Optimal solution: "
                + " | ".join(
                    [
                        f"{value[0]}: {value[1]}"
                        for value in list(zip(self.dv_names, result["opt"]["solution"]))
                    ]
                )
            )
            print(f'Optimal fitness: {result["opt"]["fitness"]}')
        print(30 * "=")

    def close_out_file(self):
        if self.out_file and not self.out_file.closed:
            self.out_file.close()

    def __del__(self):
        self.close_out_file()

    def reset(self, seed=None):
        self.output_directory = (
            f"{self.output_dir_base}/seed-{seed}" if seed else self.output_dir_base
        )

        self.close_out_file()
        self.print_headers()
        self.out_file = self.initialize_out_file()
