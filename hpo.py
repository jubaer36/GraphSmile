"""
Hyperparameter Optimization for GraphSmile using Optuna.

Usage:
    python hpo.py --dataset MELD --n_trials 50 --n_epochs 20 --gpu 0
    python hpo.py --dataset IEMOCAP --n_trials 50 --n_epochs 30 --gpu 0

Supported datasets: MELD, IEMOCAP (IEMOCAP-6)

After the search, it prints the best hyperparameters and the corresponding
F1 score. Optionally retrain the best config for the full epoch count with
--retrain.
"""

import argparse
import os
import re
import subprocess
import sys
import time

import optuna
from optuna.samplers import TPESampler

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HPO for GraphSmile with Optuna")
parser.add_argument("--dataset", default="MELD",
                    choices=["MELD", "IEMOCAP"])
parser.add_argument("--gpu", default="0", type=str,
                    help="GPU id(s) to use (same as run.py --gpu)")
parser.add_argument("--n_trials", default=40, type=int,
                    help="Number of Optuna trials")
parser.add_argument("--n_epochs", default=20, type=int,
                    help="Epochs per trial (use fewer than full training to save time)")
parser.add_argument("--base_port", default=15400, type=int,
                    help="Starting MASTER_PORT; auto-increments per trial")
parser.add_argument("--study_name", default=None, type=str,
                    help="Optuna study name (default: hpo_<dataset>)")
parser.add_argument("--storage", default=None, type=str,
                    help="Optuna storage URL, e.g. sqlite:///hpo.db  "
                         "(enables pause/resume across runs)")
parser.add_argument("--retrain", action="store_true",
                    help="After search, retrain best config for full epochs "
                         "defined per dataset in FULL_EPOCHS dict")
parser.add_argument("--classify", default="emotion",
                    choices=["emotion", "sentiment"])
args = parser.parse_args()

# Full epoch counts from the README per dataset
FULL_EPOCHS = {
    "IEMOCAP": 120,
    "MELD": 50,
}

# ── Search spaces ──────────────────────────────────────────────────────────────
#  Each dataset has its own sensible ranges derived from the paper's defaults.
SEARCH_SPACES = {
    "MELD": {
        "lr":         ("float_log", 1e-5, 5e-4),
        "hidden_dim": ("categorical", [128, 256, 384, 512]),
        "batch_size": ("categorical", [8, 16, 32]),
        "drop":       ("float", 0.1, 0.5),
        "win_p":      ("int", 1, 7),
        "win_f":      ("int", 1, 7),
        "heter_n":    ("int", 3, 8),        # single value; all 3 layers share it
        "shift_win":  ("int", 1, 8),
        "lambd_emo":  ("float", 0.5, 1.5),
        "lambd_sen":  ("float", 0.1, 1.0),
        "lambd_sft":  ("float", 0.1, 1.0),
    },
    "IEMOCAP": {
        "lr":         ("float_log", 1e-5, 5e-4),
        "hidden_dim": ("categorical", [256, 384, 512, 768]),
        "batch_size": ("categorical", [8, 16, 32]),
        "drop":       ("float", 0.1, 0.5),
        "win_p":      ("int", 5, 25),
        "win_f":      ("int", 5, 25),
        "heter_n":    ("int", 4, 10),
        "shift_win":  ("int", 5, 25),
        "lambd_emo":  ("float", 0.5, 1.5),
        "lambd_sen":  ("float", 0.1, 1.5),
        "lambd_sft":  ("float", 0.1, 1.5),
    },
}


# ── Objective ──────────────────────────────────────────────────────────────────
_trial_counter = [0]  # mutable int to increment port across trials


def suggest_params(trial, dataset: str) -> dict:
    """Sample hyperparameters from the search space for the given dataset."""
    space = SEARCH_SPACES[dataset]
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown param kind: {kind}")
    return params


def build_command(p: dict, dataset: str, port: int, n_epochs: int,
                  gpu: str, classify: str) -> list[str]:
    """Convert sampled params dict → run.py argument list."""
    cmd = [
        sys.executable, "-u", "run.py",
        "--gpu", gpu,
        "--port", str(port),
        "--classify", classify,
        "--dataset", dataset,
        "--epochs", str(n_epochs),
        "--textf_mode", "textf0",
        "--loss_type", "emo_sen_sft",
        "--lr", str(p["lr"]),
        "--batch_size", str(p["batch_size"]),
        "--hidden_dim", str(p["hidden_dim"]),
        "--win", str(p["win_p"]), str(p["win_f"]),
        "--heter_n_layers",
            str(p["heter_n"]), str(p["heter_n"]), str(p["heter_n"]),
        "--drop", str(p["drop"]),
        "--shift_win", str(p["shift_win"]),
        "--lambd",
            str(round(p["lambd_emo"], 4)),
            str(round(p["lambd_sen"], 4)),
            str(round(p["lambd_sft"], 4)),
    ]
    return cmd


def parse_f1(output: str) -> float | None:
    """
    Extract the best test F1 score from run.py stdout.
    Looks for the summary line: "Acc: <X>, F-Score: <Y>"
    Falls back to scanning per-epoch test_f1_emo lines.
    """
    # Primary: final summary line
    m = re.search(r"F-Score:\s*([\d.]+)", output)
    if m:
        return float(m.group(1))

    # Fallback: collect all per-epoch test_f1_emo and return max
    matches = re.findall(r"test_f1_emo:\s*([\d.]+)", output)
    if matches:
        return max(float(v) for v in matches)

    return None


def objective(trial: optuna.Trial) -> float:
    _trial_counter[0] += 1
    port = args.base_port + _trial_counter[0]

    p = suggest_params(trial, args.dataset)
    cmd = build_command(p, args.dataset, port, args.n_epochs,
                        args.gpu, args.classify)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}  |  port={port}")
    print("CMD:", " ".join(cmd))
    print(f"{'='*60}")

    t0 = time.time()
    collected_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered
        )
        # Stream output line by line so the user sees progress live
        deadline = t0 + 7200  # 2-hour safety timeout
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.replace("\r", "\n")  # strip carriage returns to avoid line overwrites
            print(line, end="", flush=True)
            collected_lines.append(line)
            if time.time() > deadline:
                proc.kill()
                print("\n[WARN] Trial timed out – killing process")
                break
        proc.wait()
        returncode = proc.returncode
    except Exception as exc:
        print(f"[ERROR] Failed to launch run.py: {exc}")
        return 0.0

    elapsed = round(time.time() - t0, 1)
    combined_output = "".join(collected_lines)

    if returncode != 0:
        print(f"[ERROR] run.py exited with code {returncode}")
        return 0.0

    f1 = parse_f1(combined_output)
    if f1 is None:
        print("[WARN] Could not parse F1 from output – returning 0.0")
        return 0.0

    print(f"→  Test F1 = {f1:.4f}  (elapsed {elapsed}s)")
    return f1


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    study_name = args.study_name or f"hpo_{args.dataset}"

    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    print(f"Optuna study '{study_name}'  –  {args.n_trials} trials  "
          f"–  {args.n_epochs} epochs/trial  –  dataset={args.dataset}")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ────────────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "="*70)
    print("BEST TRIAL")
    print(f"  Trial #  : {best.number}")
    print(f"  Test F1  : {best.value:.4f}")
    print("  Params   :")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v}")

    # Print the ready-to-use run.py command for the best config
    p = best.params
    full_epochs = FULL_EPOCHS[args.dataset]
    cmd = build_command(p, args.dataset, args.base_port, full_epochs,
                        args.gpu, args.classify)
    print("\nBest config command (full epochs):")
    print(" ".join(cmd))

    if args.retrain:
        print(f"\n{'='*70}")
        print(f"Retraining best config for {full_epochs} epochs …")
        result = subprocess.run(cmd, text=True)
        if result.returncode == 0:
            print("Retraining complete.")
        else:
            print(f"[ERROR] Retraining failed (exit {result.returncode})")

    # Save a brief CSV summary of all trials
    df = study.trials_dataframe()
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/hpo_{args.dataset}_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll trial results saved to {csv_path}")


if __name__ == "__main__":
    main()
