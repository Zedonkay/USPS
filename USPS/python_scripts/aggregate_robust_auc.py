import os
import json
import glob
import argparse
import numpy as np


def load_results(test_dirs, perturb_param):
    """
    For each perturbation value v, aggregate episode returns
    across all seeds (test_dirs).

    Returns:
        v_to_returns: dict[float -> list[float]]
    """
    v_to_returns = {}

    for d in test_dirs:
        pattern = os.path.join(d, f"{perturb_param}-constant-*.json")
        print(f"Loading results from: {pattern}")
        for path in glob.glob(pattern):
            with open(path, "r") as f:
                data = json.load(f)
            print(f"  Loaded: {path}")
            spec = data["perturb_spec"]
            if spec["param"] != perturb_param:
                continue

            # In your test.py, 'start' is the value v in [v_min, v_max]
            v = float(spec["start"])
            returns = data["episode_rewards"]

            v_to_returns.setdefault(v, []).extend(returns)
    print(v_to_returns)

    if not v_to_returns:
        raise RuntimeError(
            f"No JSON files found for param={perturb_param} in: {test_dirs}"
        )
    return v_to_returns


def compute_robust_auc(v_to_returns):
    """
    Implements the Robust-AUC described in C.3:

      - For each v, r_0.10(v) = 10% quantile of returns
      - AUC_0.10 = integral of r_0.10(v) over v (trapezoidal rule)
      - Robust-AUC = AUC_0.10 / (v_max - v_min)

      - Similarly compute AUC_0.05 and AUC_0.15
      - Uncertainty = (AUC_0.15 - AUC_0.05) / (v_max - v_min)
    """
    # Sort by v
    items = sorted(v_to_returns.items(), key=lambda x: x[0])
    vs = np.array([v for v, _ in items])
    print(vs)

    # Collect quantiles per v over *all* episodes from all seeds
    r05 = np.array([np.quantile(rets, 0.05) for _, rets in items])
    r10 = np.array([np.quantile(rets, 0.10) for _, rets in items])
    r15 = np.array([np.quantile(rets, 0.15) for _, rets in items])

    vmin, vmax = vs.min(), vs.max()
    interval = vmax - vmin

    # Trapezoidal integrals
    area05 = np.trapz(r05, vs)
    area10 = np.trapz(r10, vs)
    area15 = np.trapz(r15, vs)

    robust_auc = area10 / interval
    uncertainty = (area15 - area05) / interval

    return robust_auc, uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dirs",
        type=str,
        required=True,
        help="Comma-separated list of EXPERIMENTS_DIR (one per seed). "
             "Each must contain a 'test/' subdir with JSON files.",
    )
    parser.add_argument(
        "--perturb_param",
        type=str,
        required=True,
        help="Environmental variable name, e.g. pole_length, pole_mass, ...",
    )
    args = parser.parse_args()

    exp_dirs = [p.strip() for p in args.exp_dirs.split(",") if p.strip()]
    test_dirs = [os.path.join(p, "test") for p in exp_dirs]

    v_to_returns = load_results(test_dirs, args.perturb_param)
    robust_auc, uncertainty = compute_robust_auc(v_to_returns)

    print(f"Perturb param: {args.perturb_param}")
    print(f"Robust-AUC: {robust_auc:.2f} ({uncertainty:.2f})")


if __name__ == "__main__":
    main()
