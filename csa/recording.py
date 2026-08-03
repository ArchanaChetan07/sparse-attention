"""Recording schema discipline (proposal Appendix B).

Aggregate online, never dump raw attention tensors. Each recorded row is one
decode step with layer-aggregated scalars; request metadata is joined on.
Every result file is accompanied by a machine fingerprint.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


def machine_fingerprint() -> dict:
    fp = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
    }
    try:
        import torch
        fp["torch"] = torch.__version__
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            fp["gpu_mem_gb"] = round(props.total_memory / 2**30, 1)
            fp["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import transformers
        fp["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,power.limit,pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if q.returncode == 0:
            drv, plim, gen, width = [s.strip() for s in q.stdout.strip().split(",")]
            fp.update({"driver": drv, "power_limit": plim,
                       "pcie": f"gen{gen} x{width}"})
    except Exception:
        pass
    return fp


def save_results(rows: list[dict], meta: dict, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    with open(out / f"{name}.meta.json", "w") as f:
        json.dump({"meta": meta, "fingerprint": machine_fingerprint()}, f, indent=2)
    return path
