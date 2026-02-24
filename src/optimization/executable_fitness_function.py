import subprocess
from pathlib import Path


class ExecutableFitnessFunction:
    def __init__(self, executable_path, timeout=30):
        self.executable_path = Path(executable_path).resolve()
        self.timeout = timeout
        self.work_dir = self.executable_path.parent
    
    def __call__(self, *args, **kwargs):
        # Write input file
        task_file = self.work_dir / 'task.dat'
        with open(task_file, 'w') as f:
            f.write(f"{len(args)}\n")
            for value in args:
                f.write(f"{value}\n")
        
        # Execute
        try:
            subprocess.run(
                [str(self.executable_path)],
                cwd=self.work_dir,
                timeout=self.timeout,
                check=True
            )
        except subprocess.TimeoutExpired:
            return float('inf')
        
        # Read output file
        result_file = self.work_dir / 'task.res'
        with open(result_file, 'r') as f:
            objectives = [float(line.strip()) for line in f if line.strip()]
        
        return objectives[0] if len(objectives) == 1 else objectives
