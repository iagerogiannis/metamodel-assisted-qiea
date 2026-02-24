import pandas as pd
import numpy as np


class InMemoryDB:
    def __init__(self):
        self.data = {}

    def get_by_id(self, id):
        if id not in self.data:
            return None
        return self.data[id]["output"]

    def put(self, id, gen, input_array, output):
        self.data[id] = {
            "gen": int(gen),
            "input": np.array(input_array).tolist(),
            "output": output.tolist() if isinstance(output, np.ndarray) else output,
        }

    def get_all(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame(columns=["id", "gen", "input", "output"])

        records = []
        for id, record in self.data.items():
            records.append(
                {
                    "id": id,
                    "gen": record["gen"],
                    "input": record["input"],
                    "output": record["output"],
                }
            )

        return pd.DataFrame(records)

    def clear(self):
        self.data.clear()
        print("Cleared in-memory database")
