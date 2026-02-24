import os
import pandas as pd


def generate_csv_from_log(dir_path):
    """Generate out.csv from EA_L1.log if it doesn't exist. Returns True if csv is available."""
    out_csv = os.path.join(dir_path, "out.csv")
    if os.path.exists(out_csv):
        return True

    log_file = os.path.join(dir_path, "EA_L1.log")
    if not os.path.exists(log_file):
        return False

    rows = []
    with open(log_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            rows.append([parts[1], parts[3], parts[14]])

    df = pd.DataFrame(rows, columns=["Gen Index", "Num of Evals", "Fitness Score"])
    df.to_csv(out_csv, index=False)
    return True
