"""Run the whole project from raw data to results.

    python run_pipeline.py              # everything (download is skipped if files exist)
    python run_pipeline.py --skip-download
    python run_pipeline.py --quick      # fewer permutations / stability rounds
"""
import argparse
import time

from src import (data_acquisition, experiment_comparison, experiments,
                 feature_engineering, preprocessing, renewable_analysis,
                 reporting, visualization)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = [] if a.skip_download else [("acquisition", data_acquisition.main)]
    steps += [("preprocessing", preprocessing.main),
              ("feature engineering", feature_engineering.main),
              ("renewable metrics", renewable_analysis.main),
              ("experiments", (lambda: experiments.main(99, 5)) if a.quick
               else experiments.main),
              ("experiment comparison", experiment_comparison.main),
              ("figures", visualization.main),
              ("results report", reporting.main)]
    for name, fn in steps:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        fn()
        print(f"--- {name} done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
