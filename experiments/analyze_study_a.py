"""Re-run Study A analysis over any results directory.

Sweeps are expensive and are run at different times; every run must be
analysed by the same code for cross-run comparisons to mean anything.
Rewrites summary.json and figures/ in place.

    python experiments/analyze_study_a.py results/study_a_0.5b results/study_a_1.5b
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csa.analysis import analyze_dir

KEYS = ["teacher_steps", "teacher_flip_rate", "dense_qa_accuracy",
        "h4_all_requests", "h4_answerable", "h4_answerable_within_budget",
        "h4_floor_effect"]


def main():
    dirs = sys.argv[1:] or ["results/study_a_0.5b", "results/study_a_1.5b"]
    for d in dirs:
        p = Path(d)
        if not (p / "steps.csv").exists():
            print(f"skip {d}: no steps.csv")
            continue
        s = analyze_dir(p)
        print(f"\n===== {d} =====")
        print("best label-free AUC:",
              max((v, k) for k, v in s["auc_teacher_flip"].items()
                  if not k.startswith("ORACLE")))
        for k in KEYS:
            if k in s:
                print(f"{k}: {json.dumps(s[k], default=float)[:400]}")


if __name__ == "__main__":
    main()
