import time
import statistics

from step_2_process_polars import run_polars
from step_2_process_pandas import run_pandas

PARQUET_PATH = "/Users/spicy.kev/Library/CloudStorage/GoogleDrive-kspicer@stfrancis.edu/My Drive/dh_stuff/projects/fourth_down_nfl_project/data/pbp_raw.parquet"
RUNS = 5

def benchmark(func, label):
    times = []

    for _ in range(RUNS):
        start = time.perf_counter()
        result = func(PARQUET_PATH)
        end = time.perf_counter()

        # force materialization
        _ = result.shape

        times.append(end - start)

    print(f"{label}")
    print(f"  runs: {RUNS}")
    print(f"  avg:  {statistics.mean(times):.3f}s")
    print(f"  min:  {min(times):.3f}s")
    print(f"  max:  {max(times):.3f}s")
    print()

if __name__ == "__main__":
    benchmark(run_pandas, "Pandas")
    benchmark(run_polars, "Polars")
