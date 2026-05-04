import sys
import os
import csv
import time
import signal

import sympy as sp

from buchberger_impl import (
    buchberger,
    load_systems_grouped,
    TimeoutException,
)


# ---- CLI arguments ----
if len(sys.argv) < 4:
    print("Usage: python benchmark_random.py STRATEGY OUTPUT_CSV INPUT_CSV")
    print("  STRATEGY:    sugar | degree | normal | s_poly_entropy")
    print("  OUTPUT_CSV:  path to write results to")
    print("  INPUT_CSV:   path to dataset CSV")
    sys.exit(1)

strategy    = sys.argv[1]
output_path = sys.argv[2]
input_csv   = sys.argv[3]


if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    R, _ = sp.xring("x0,x1,x2", sp.FF(32003), "grevlex")
    timeout_seconds = 10 * 60
    run_records = []
    timed_out_systems = []

    systems = load_systems_grouped(input_csv, R, has_header=True)
    print(f"Processing {input_csv} with strategy={strategy}")
    print(f"Loaded {len(systems)} systems")

    for system_id, system_polys in systems.items():
        print(f"  System {system_id}")
        start = time.time()
        finished = 0
        elapsed = timeout_seconds
        basis_size = 0
        n_steps = 0

        try:
            signal.alarm(timeout_seconds)
            basis, n_steps = buchberger(system_polys, selection=strategy)
            signal.alarm(0)
            elapsed = time.time() - start
            finished = 1
            basis_size = len(basis)
        except TimeoutException:
            signal.alarm(0)
            elapsed = time.time() - start
            finished = 0
            timed_out_systems.append((input_csv, system_id))

        run_records.append({
            "source_file": input_csv,
            "system_id": system_id,
            "strategy": strategy,
            "elapsed": elapsed,
            "basis_size": basis_size,
            "n_steps": n_steps,
            "reward": None,
            "finished": finished,
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_file", "system_id", "strategy",
            "elapsed", "basis_size", "n_steps", "reward", "finished"
        ])
        writer.writeheader()
        writer.writerows(run_records)

    print(f"Wrote {len(run_records)} records to {output_path}")
    if run_records:
        print(f"Success rate: {sum(r['finished'] for r in run_records)/len(run_records):.1%}")
    print(f"Timed out: {len(timed_out_systems)} systems")