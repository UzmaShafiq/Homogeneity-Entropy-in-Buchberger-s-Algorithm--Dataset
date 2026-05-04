# ============================================================
# Benchmark PHCpack
# ============================================================
import sys
import time
import signal
from buchberger_impl import buchberger, TimeoutException
from phc_parser import PHCRingLoader, FixedPHCPolyLoader, load_smart_rings, load_gf_rings
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python phc_parser.py STRATEGY OUTPUT_CSV INPUT_CSV [FIELD] [MODULUS]")
        print("  STRATEGY:    sugar | degree | normal | s_poly_entropy")
        print("  OUTPUT_CSV:  path to write results to")
        print("  INPUT_CSV:   PHCpack-format CSV")
        print("  FIELD:       Q (default) | GF")
        print("  MODULUS:     prime modulus (required if FIELD=GF, e.g. 32003)")
        sys.exit(1)

    import os

    strategy    = sys.argv[1]
    output_path = sys.argv[2]
    input_csv   = sys.argv[3]
    field       = sys.argv[4] if len(sys.argv) > 4 else "Q"
    modulus     = int(sys.argv[5]) if len(sys.argv) > 5 else None

    if field == "GF" and modulus is None:
        print("ERROR: MODULUS required when FIELD=GF")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading {input_csv} (field={field}"
          + (f", modulus={modulus}" if modulus else "") + ")...")
    if field == "GF":
        rings = PHCRingLoader(input_csv).load_all_rings(field="GF", modulus=modulus)
    else:
        rings = load_smart_rings(input_csv)

    loader = FixedPHCPolyLoader(input_csv, rings)
    polys_all = loader.load_all_polys(verbose=False)

    if loader.failure_report():
        print(f"WARNING: {len(loader.failure_report())} polynomials failed to parse.")

    print(f"Loaded {len(polys_all)} systems")

    timeout_seconds = 10 * 60
    run_records = []
    timed_out_systems = []

    for system_id, system_polys in sorted(polys_all.items()):
        print(f"  Sys{system_id} ({len(system_polys)} eqs)", end=" ", flush=True)
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
            print(f"-> DONE | basis_size={basis_size} | steps={n_steps} | time={elapsed:.2f}s")
        except TimeoutException:
            signal.alarm(0)
            elapsed = time.time() - start
            finished = 0
            timed_out_systems.append((input_csv, system_id))
            print(f"-> TIMEOUT after {elapsed:.2f}s")
        except Exception as e:
            signal.alarm(0)
            elapsed = time.time() - start
            finished = -1
            print(f"-> ERROR: {type(e).__name__}: {e}")

        run_records.append({
            "source_file": input_csv,
            "system_id":   system_id,
            "nvars":       rings[system_id].n_variables,
            "neqs":        len(system_polys),
            "domain":      rings[system_id].field,
            "strategy":    strategy,
            "elapsed":     elapsed,
            "basis_size":  basis_size,
            "n_steps":     n_steps,
            "reward":      None,
            "finished":    finished,
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_file", "system_id", "nvars", "neqs", "domain",
            "strategy", "elapsed", "basis_size", "n_steps", "reward", "finished"
        ])
        writer.writeheader()
        writer.writerows(run_records)

    print(f"\nWrote {len(run_records)} records to {output_path}")
    if run_records:
        success = sum(r["finished"] == 1 for r in run_records)
        errors  = sum(r["finished"] == -1 for r in run_records)
        print(f"Success:  {success}/{len(run_records)} ({success/len(run_records):.1%})")
        print(f"Errors:   {errors}")
        print(f"Timeouts: {len(timed_out_systems)}")