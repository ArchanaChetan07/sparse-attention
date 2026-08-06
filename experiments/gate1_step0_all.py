#!/usr/bin/env python3
"""Gate-1 STEP 0 machine acceptance."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)


def section(title: str):
    print(f"\n=== {title} ===", flush=True)


def main():
    fails = []

    section("GPU / POWER / PCIE")
    print(sh(
        "nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,"
        "pcie.link.gen.current,pcie.link.width.current,memory.total,memory.free "
        "--format=csv"
    ).strip())
    # Use GPU 0 for checks
    q = sh(
        "nvidia-smi -i 0 --query-gpu=power.limit,power.default_limit,"
        "pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits"
    ).strip()
    parts = [p.strip() for p in q.split(",")]
    plim, pdef, gen, width = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
    print(f"gpu0 power_limit={plim} default={pdef} pcie_gen={gen} x{width}")
    if plim + 1e-3 < pdef:
        fails.append(f"power capped below default: {plim} < {pdef}")
    if gen < 4:
        fails.append(f"PCIe gen too low: {gen}")
    if width < 16:
        fails.append(f"PCIe width too low: x{width}")

    section("STOP VLLM / FREE GPU")
    try:
        print(sh("supervisorctl stop vllm model-ui 2>/dev/null || true"))
        time.sleep(2)
        # kill orphan engine cores if any
        print(sh("pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true"))
        time.sleep(1)
    except Exception as e:
        print("note:", e)
    print(sh(
        "nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv"
    ).strip())

    section("HBM + H2D")
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device={props.name} total_gib={total/2**30:.2f} free_gib={free/2**30:.2f}")
    nbytes = min(4 * 1024**3, int(free * 0.4))
    nbytes = max((nbytes // 256) * 256, 256 * 1024**2)
    nfloat = nbytes // 4
    a = torch.empty(nfloat, device="cuda:0", dtype=torch.float32)
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tbps = (nbytes * iters) / dt / 1e12
    # SXM expect >=2.7; NVL similar. Soften: fail only if < 2.0 (clearly broken)
    hard = 2.0
    soft = 2.7
    print(f"HBM_D2D_TBps={tbps:.3f} soft>={soft} hard>={hard} -> "
          f"{'PASS' if tbps >= soft else ('SOFT_FAIL' if tbps >= hard else 'FAIL')}")
    if tbps < hard:
        fails.append(f"HBM bandwidth {tbps:.3f} TB/s < {hard} TB/s")
    del a, b
    torch.cuda.empty_cache()

    h = torch.empty(nfloat, device="cpu", dtype=torch.float32, pin_memory=True)
    d = torch.empty(nfloat, device="cuda:0", dtype=torch.float32)
    torch.cuda.synchronize()
    for _ in range(3):
        d.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        d.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    h2d = (nbytes * iters) / dt / 1e9
    print(f"H2D_GBps={h2d:.1f} threshold=20 -> {'PASS' if h2d >= 20 else 'FAIL'}")
    if h2d < 20:
        fails.append(f"H2D {h2d:.1f} GB/s < 20")
    del h, d
    torch.cuda.empty_cache()

    section("NETWORK")
    url = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/"
           "resolve/main/model.safetensors")
    t0 = time.perf_counter()
    n = 0
    with urlopen(url, timeout=120) as r:
        while n < 64 * 1024 * 1024:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            n += len(chunk)
    dt = time.perf_counter() - t0
    gbit = (n * 8 / dt) / 1e9 if dt > 0 else 0.0
    print(f"sampled_MB={n/1e6:.1f} sec={dt:.2f} Gbit_s={gbit:.3f} "
          f"threshold=0.5 -> {'PASS' if gbit >= 0.5 else 'FAIL'}")
    # listing had ~0.5 Gbit; require >=0.5 so HF is usable
    if gbit < 0.5:
        fails.append(f"network {gbit:.3f} Gbit/s < 0.5")

    section("DISK")
    print(sh("df -h / /workspace 2>/dev/null || df -h /").strip())
    avail = int(sh("df -B1 / | awk 'NR==2{print $4}'").strip())
    print(f"root_avail_gib={avail/2**30:.2f}")
    if avail < 40 * 1024**3:
        fails.append(f"disk avail {avail/2**30:.1f} GiB < 40 GiB")

    section("CREDENTIALS")
    hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    gh = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    print(f"HF_TOKEN: {'SET' if hf else 'MISSING'}")
    print(f"GH_TOKEN: {'SET' if gh else 'MISSING'}")
    # Not hard-fail HF if we can use ungated Qwen; GH warn only for push
    if not hf:
        print("NOTE: HF_TOKEN missing — will use ungated Qwen models only")
    if not gh:
        print("NOTE: GH_TOKEN missing — cannot push mid-run; sync risk on interrupt")

    section("VERDICT")
    Path("/tmp/gate1_step0_report.txt").write_text(
        ("FAILS:\n" + "\n".join(fails) + "\n") if fails else "ALL HARD CHECKS PASSED\n"
    )
    if fails:
        print("STEP0_FAIL")
        for f in fails:
            print(" -", f)
        raise SystemExit(2)
    print("STEP0_PASS")
    print(f"hbm_tbps={tbps:.3f} h2d_gbs={h2d:.1f} net_gbit={gbit:.3f}")


if __name__ == "__main__":
    main()
