# Metamodel Assisted Quantum Inspired Evolutionary Algorithms

Welcome to the Metamodel Assisted Quantum Inspired Evolutionary Algorithms repository! This repository contains an implementation of quantum-inspired evolutionary algorithms, which are a class of optimization algorithms inspired by principles from quantum mechanics. Metamodel assistance is used to enhance the performance of these algorithms.

## Features

- Implementation of various quantum-inspired evolutionary algorithms
- Metamodel-assisted optimization techniques for improved performance
- Support for different types of optimization problems
- Customizable parameters for fine-tuning the algorithms
- Documentation and examples to help you get started

## Getting Started

To get started with the Quantum-Inspired Evolutionary Algorithms, follow these steps:

1. Clone the repository: `git clone https://github.com/iagerogiannis/metamodel-assisted-qiea`
2. Navigate to the project directory: `cd metamodel-assisted-qiea`
3. Set up a virtual environment:
   - On Windows, run: `py -m venv .venv`
   - On Unix or MacOS, run: `python3 -m venv .venv`
4. Activate the virtual environment:
   - On Windows, run: `.\.venv\Scripts\activate`
   - On Unix or MacOS, run: `source .venv/bin/activate`
5. Install the required dependencies:
   - On Windows, run: `pip install -r requirements-win.txt`
   - On Unix or MacOS, run: `pip install -r requirements-unix.txt`
6. Explore the examples and documentation to understand how to use the algorithms
7. Start experimenting with the algorithms on your own optimization problems

## Usage

Run experiments via `main.py`. One of three mutually exclusive input modes is required:

### Run built-in examples

```bash
python main.py --run-examples
```

Runs all configs found in the `config_examples/` directory and saves visualizations to `output/plots/`.

If you want to write your own configuration files, refer to the following examples to understand the expected structure and format:

- `config_examples/qea-ackley.json` — reference config for a standard QEA
- `config_examples/maqea-ackley.json` — reference config for a metamodel-assisted QEA
- `config_examples/metamodels/config-100.json` — reference config for the metamodel

### Run a single config file

```bash
python main.py --config path/to/config.json
```

### Run all configs in a directory

```bash
python main.py --config-dir path/to/configs/
```

### Optional flags

| Flag | Description |
|---|---|
| `--output-dir DIR` | Directory to save plots and results (default: `output/plots`) |
| `--no-visualize` | Skip visualization and only run experiments |
| `--filter SUBSTRING` | Only run experiments whose name contains the given substring |

### Examples

```bash
# Run a specific config and save results to a custom directory
python main.py --config configs/my_experiment.json --output-dir results/

# Run all configs in a directory, filtering by name
python main.py --config-dir configs/ --filter sphere

# Run all configs without generating plots
python main.py --config-dir configs/ --no-visualize
```

## Contributing

Contributions are welcome! If you have any ideas, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

Happy optimizing!