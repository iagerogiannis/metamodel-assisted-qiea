import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import atexit

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class ParquetDB:
    def __init__(self, directory: str, multi_objective: bool = False):
        self.dataset_path = Path(directory) / "parquet_batches"
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        self.columns = ["id", "gen", "input", "output"]  # Added "id"
        self.buffer = []
        self.buffer_limit = 100
        self.multi_objective = multi_objective

        atexit.register(self._flush)

    def _buffer_record(self, id, gen, input_array, output:'np.ndarray | np.number'=None):
        input_array = np.array(input_array)
        record = {
            "id": id,
            "gen": int(gen),
            "input": input_array.tolist(),
            "output": output.tolist() if isinstance(output, np.ndarray) else output,
        }

        self.buffer.append(record)

        if len(self.buffer) >= self.buffer_limit:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer, columns=self.columns)
        if df.empty:
            self.buffer.clear()
            return

        df["input"] = df["input"].apply(lambda x: np.asarray(x, dtype=np.float64).tolist() if x is not None else None)

        def process_output(col, multi_objective):
            arr = df[col].to_numpy()
            if multi_objective:
                return [np.asarray(x, dtype=np.float64).tolist() if x is not None else None for x in arr]
            else:
                return [float(x[0]) if isinstance(x, list) and x else float(x) if x is not None else None for x in arr]

        df["output"] = process_output("output", self.multi_objective)

        if self.multi_objective:
            schema = pa.schema([
                pa.field("id", pa.string(), nullable=False),
                pa.field("gen", pa.int64(), nullable=False),
                pa.field("input", pa.list_(pa.float64()), nullable=False),
                pa.field("output", pa.list_(pa.float64()), nullable=False),
            ])
        else:
            schema = pa.schema([
                pa.field("id", pa.string(), nullable=False),
                pa.field("gen", pa.int64(), nullable=False),
                pa.field("input", pa.list_(pa.float64()), nullable=False),
                pa.field("output", pa.float64(), nullable=False),
            ])

        filename = f"batch_{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, self.dataset_path / filename, compression="snappy")
        self.buffer.clear()

    def get_all(self) -> pd.DataFrame:
        self._flush()
        dataset = pq.ParquetDataset(self.dataset_path)
        return dataset.read_pandas().to_pandas()

    def get_by_id(self, id):
        df = self.get_all()

        if df.empty:
            return None

        item = df[df["id"] == id][["input", "output"]]

        if item.empty:
            return None

        return item.iloc[0].to_dict()['output']

    def put(self, id, gen, input_array: np.ndarray, output: 'np.ndarray | np.number'):
        self._buffer_record(id, gen, input_array, output=output)

    def clear(self):
        for file in self.dataset_path.glob("*.parquet"):
            file.unlink()
        self.buffer.clear()
        self._flush()
        print(f"Cleared all parquet data in {self.dataset_path}")

    def __del__(self):
        try:
            self._flush()
        except Exception:
            pass
