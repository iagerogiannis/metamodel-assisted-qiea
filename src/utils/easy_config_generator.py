"""
EASY Configuration File Generator

Generates configuration files for the EASY optimization framework
based on optimization parameters.
"""

from pathlib import Path
import pkg_resources


class EASYConfigGenerator:
    def __init__(
        self,
        template_path="src/config_templates/easy_template.ini",
        output_base_dir="easy-setup/cases",
    ):
        self.template_path = template_path
        self.template = self._load_template()
        self.output_base_dir = Path(output_base_dir)

    def _load_template(self):
        """Load the template file"""
        template_dir = Path(self.template_path).parent
        template_dir.mkdir(parents=True, exist_ok=True)

        # Create default template if it doesn't exist
        if not Path(self.template_path).exists():
            self._create_default_template()

        with open(self.template_path, "r") as f:
            return f.read()

    def _create_default_template(self):
        """Create a default template file by reading from the embedded template"""
        # Read template from package data
        try:
            template_content = pkg_resources.resource_string(
                __name__, "easy_template.ini"
            ).decode("utf-8")
        except:
            # Fallback: read from file in same directory
            template_file = Path(__file__).parent / "easy_template.ini"
            if template_file.exists():
                with open(template_file, "r") as f:
                    template_content = f.read()
            else:
                raise FileNotFoundError(
                    f"Template file not found. Please ensure 'easy_template.ini' "
                    f"exists in the same directory as this script or in config_templates/"
                )

        with open(self.template_path, "w") as f:
            f.write(template_content)

    def _generate_default_path(self, params):
        """Generate a default output path based on parameters"""
        test_function = params.get("test_function", "unknown")
        mu = params.get("mu", params.get("population_size", 20))
        lambda_val = params.get("lambda", mu * 3)
        elitism_rate = params.get("elitism_rate", 0.1)
        xover_rate = params.get("crossover_rate", 0.9)
        muta_rate = params.get("mutation_rate", 0.02)
        seed = params.get("random_seed", 42)

        case_name = f"{test_function}/mu{mu}_lambda{lambda_val}_elite{elitism_rate}_xover{xover_rate}_mrate{muta_rate}"
        seed_dir = f"seed-{seed}"

        output_path = self.output_base_dir / case_name / seed_dir / "config.easy"
        return str(output_path)

    def generate_config(self, params, output_path=None):
        """
        Generate configuration file from parameters

        Args:
            params: Dictionary containing optimization parameters
            output_path: Path where to save the generated config file.
                        If None, generates a default path based on parameters.
        """
        # Generate default output path if not provided
        if output_path is None:
            output_path = self._generate_default_path(params)

        # Build variables section
        variables_section = ""
        design_vars = params.get("design_variables", {})

        # Handle both dict format (with 'number', 'bounds', 'bits') and list format
        if isinstance(design_vars, dict):
            num_vars = design_vars.get("number", 1)
            bounds = design_vars.get("bounds", [0.0, 1.0])
            bits = design_vars.get("bits", 10)
            lower_bound, upper_bound = bounds[0], bounds[1]

            for i in range(1, num_vars + 1):
                variables_section += f"{bits} {lower_bound} {upper_bound} x_{i}\n"
        else:
            # List format (original)
            for var in design_vars:
                variables_section += f"{var['bits']} {var['lower_bound']} {var['upper_bound']} {var['name']}\n"

        # Build evaluation script line
        test_function = params.get("test_function", "rastrigin")
        eval_script_dir = params.get("eval_script_dir", "/home/user/test-functions")
        eval_script = params.get("eval_script", f"{eval_script_dir}/{test_function}")
        max_evals = params.get("max_evaluations", 6000)
        eval_script_line = f"S {eval_script} none {max_evals} 1.0 null"

        # Fill template with values
        num_objectives = params.get("num_objectives", 1)
        config_values = {
            # Driver section
            "case_init": params.get("case_init", 0),
            "restore_db": params.get("restore_db", 0),
            "num_objectives": num_objectives,
            "parallel_mode": params.get("parallel_mode", 2),
            "concurrent_procs": params.get("concurrent_procs", 1),
            "eval_stats_step": params.get("eval_stats_step", 500),
            "save_db_step": params.get("save_db_step", 0),
            "num_levels": params.get("num_levels", 1),
            "num_params": params.get("num_params", 1),
            "log_file": params.get("log_file", "EA"),
            "state_file": params.get("state_file", "state"),
            "solution_file": params.get("solution_file", "out"),
            "db_file": params.get("db_file", "DB"),
            # Evaluators section
            "num_eval_scripts": params.get("num_eval_scripts", 1),
            "eval_script_line": eval_script_line,
            # Level 1 section
            "level_type": params.get("level_type", 0),
            "param_id": params.get("param_id", 1),
            "num_demes": params.get("num_demes", 1),
            "gen_save_step": params.get("gen_save_step", 2),
            "cont_save_step": params.get("cont_save_step", 0),
            "random_seed": params.get("random_seed", 1982),
            # Migration parameters
            "migration_freq": params.get("migration_freq", 1),
            "max_migs": params.get("max_migs", 0),
            "emigrants_best": params.get("emigrants_best", 0),
            "emigrants_random": params.get("emigrants_random", 0),
            "num_immigrants": params.get("num_immigrants", 0),
            "migration_replace": params.get("migration_replace", 1),
            "migration_graph": params.get("migration_graph", 1),
            "infection_freq": params.get("infection_freq", 1),
            "max_infections": params.get("max_infections", 0),
            "infection_muta_mult": params.get("infection_muta_mult", 1.0),
            "infection_max_muta": params.get("infection_max_muta", 1.0),
            "infection_radius": params.get("infection_radius", 1.0),
            "infection_duration": params.get("infection_duration", 1),
            "max_sharing_penalty": params.get("max_sharing_penalty", 100.0),
            # Hierarchical parameters
            "level_mig_higher_first": params.get("level_mig_higher_first", 1),
            "level_mig_higher_freq": params.get("level_mig_higher_freq", 1),
            "level_mig_lower_first": params.get("level_mig_lower_first", 1),
            "level_mig_lower_freq": params.get("level_mig_lower_freq", 1),
            "import_elites_higher_first": params.get("import_elites_higher_first", 1),
            "import_elites_higher_freq": params.get("import_elites_higher_freq", 1),
            "import_elites_lower_first": params.get("import_elites_lower_first", 1),
            "import_elites_lower_freq": params.get("import_elites_lower_freq", 1),
            "reeval_higher": params.get("reeval_higher", 1.0),
            "reeval_lower": params.get("reeval_lower", 1.0),
            "tol_inefficient_migs": params.get("tol_inefficient_migs", 5),
            # Population properties
            "init_mode": params.get("init_mode", 0),
            "num_scripts": params.get("num_scripts", 1),
            "script_ids": params.get("script_ids", "1"),
            "script_percentages": params.get("script_percentages", "1.0"),
            "coding_type": params.get("coding_type", 2),
            "es_stdev": params.get("es_stdev", 0.1),
            "mu_parents": params.get("mu", params.get("population_size", 20)),
            "lambda_offspring": params.get(
                "lambda", params.get("mu", params.get("population_size", 20)) * 3
            ),
            "kappa_lifespan": params.get("kappa_lifespan", 0),
            "rho_parents_per_offspring": params.get("rho_parents_per_offspring", 2),
            "num_elites": params.get("num_elites", 40 if num_objectives > 1 else 1),
            "allow_penalized_elites": params.get("allow_penalized_elites", 0),
            "force_elites": params.get("force_elites", 1),
            "prob_select_elite": params.get("prob_select_elite", 0.1),
            "tournament_size": params.get("tournament_size", 3),
            "tournament_prob": params.get("tournament_prob", 0.9),
            "xover_prob": params.get("xover_prob", params.get("crossover_rate", 0.9)),
            "xover_type": params.get("xover_type", 5),
            "xover_sigma_type": params.get("xover_sigma_type", 0),
            "muta_prob": params.get("muta_prob", params.get("mutation_rate", 0.02)),
            "muta_mode": params.get("muta_mode", 0),
            "muta_refinement": params.get("muta_refinement", 2.0),
            "dynamic_exploration": params.get("dynamic_exploration", 1.0),
            "dynamic_exploration_gen": params.get("dynamic_exploration_gen", 700000),
            "max_generations": params.get("max_generations", 700000),
            "max_evaluations": params.get("max_evaluations", 6000),
            "max_idle_gen": params.get("max_idle_gen", 700000),
            "max_idle_evals": params.get("max_idle_evals", 700000),
            "fitness_mode": params.get("fitness_mode", 4),
            "niche_radius": params.get("niche_radius", 0.1),
            "distance_space": params.get("distance_space", 1),
            "distance_nondim": params.get("distance_nondim", 2),
            "nested_distance_mult": params.get("nested_distance_mult", 1.2),
            # Variable adaptation
            "adaptation_freq": params.get("adaptation_freq", 0),
            "max_adaptations": params.get("max_adaptations", 0),
            "adaptation_extent_mult": params.get("adaptation_extent_mult", 1.0),
            # Metamodel parameters
            "metamodel_type": params.get("metamodel_type", 0),
            # "metamodel_type": params.get("metamodel_type", 1),  # RBF_IFs
            "preclassif_filter": params.get("preclassif_filter", 0),
            "preclassif_pop_pct": params.get("preclassif_pop_pct", 0.1),
            "trust_region": params.get("trust_region", 0.01),
            "max_offset": params.get("max_offset", 0.7),
            "correct_on": params.get("correct_on", 1),
            "exact_evals_min": params.get("exact_evals_min", 2),
            "exact_evals_max": params.get("exact_evals_max", 8),
            "idle_gen_ipe_pause": params.get("idle_gen_ipe_pause", 15),
            "extrapolate_pred": params.get("extrapolate_pred", 1),
            "nondim_pred": params.get("nondim_pred", 1),
            "min_not_failed": params.get("min_not_failed", 10),
            "failed_obj_factor": params.get("failed_obj_factor", 5.0),
            "min_db_entries": params.get("min_db_entries", 100),
            "min_not_failed_db": params.get("min_not_failed_db", 50),
            "max_db_pct": params.get("max_db_pct", 1.0),
            "use_failed_patterns": params.get("use_failed_patterns", 0),
            "train_patterns_min": params.get("train_patterns_min", 20),
            "train_patterns_max": params.get("train_patterns_max", 40),
            "proximity_factor": params.get("proximity_factor", 1.4),
            "rbf_radius": params.get("rbf_radius", 1.0),
            "importance_factors_relax": params.get("importance_factors_relax", 0.4),
            "use_pca": params.get("use_pca", 0),
            "svm_best_pct": params.get("svm_best_pct", 0.5),
            "svm_fitness_scenario": params.get("svm_fitness_scenario", 1),
            "svm_use_failed": params.get("svm_use_failed", 0),
            "grbfn_min_centers": params.get("grbfn_min_centers", 2),
            "grbfn_max_centers": params.get("grbfn_max_centers", 120),
            "grbfn_radius_mult": params.get("grbfn_radius_mult", 0.5),
            "grbfn_test_pct": params.get("grbfn_test_pct", 0.3),
            "grbfn_max_idle": params.get("grbfn_max_idle", 10),
            "grbfn_learning_rate": params.get("grbfn_learning_rate", 0.1),
            "gng_center_move": params.get("gng_center_move", 0.05),
            "gng_neigh_move": params.get("gng_neigh_move", 6.0e-4),
            "gng_max_conn_age": params.get("gng_max_conn_age", 80),
            "gng_add_step_factor": params.get("gng_add_step_factor", 10.0),
            "gng_error_decrease_interp": params.get("gng_error_decrease_interp", 0.5),
            "gng_error_decrease_gen": params.get("gng_error_decrease_gen", 0.9995),
            "gng_neigh_pct": params.get("gng_neigh_pct", 0.1),
            "gng_prob_select_neigh": params.get("gng_prob_select_neigh", 0.95),
            "gng_iter_frac_neigh": params.get("gng_iter_frac_neigh", 0.0),
            "grbfn_mode": params.get("grbfn_mode", 2),
            "grbfn_threshold_ratio": params.get("grbfn_threshold_ratio", 0.5),
            # Variables section
            "num_variables": (
                len(params["design_variables"])
                if isinstance(params.get("design_variables"), list)
                else params.get("design_variables", {}).get("number", 1)
            ),
            "variables_section": variables_section.strip(),
            # Constraints
            "num_constraints": params.get("num_constraints", 0),
            "num_param_links": params.get("num_param_links", 0),
            "constraints_aux": "".join("1.0 \n" for _ in range(num_objectives)),
        }

        # Generate config content
        config_content = self.template.format(**config_values)

        # Save to file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(config_content)

        print(f"Configuration file generated: {output_path}")
