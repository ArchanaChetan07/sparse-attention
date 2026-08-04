# Study A across two model scales on the identical task set.
# The two runs are directly comparable, which is what Ablation 5
# (detector transfer across models) requires.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Study A: Qwen2.5-0.5B-Instruct ==="
python experiments\study_a_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct `
    --per-family 2 --max-new-long 64 --out results\study_a_0.5b

Write-Host "=== Study A: Qwen2.5-1.5B-Instruct ==="
python experiments\study_a_smoke.py --model Qwen/Qwen2.5-1.5B-Instruct `
    --per-family 2 --max-new-long 64 --out results\study_a_1.5b

Write-Host "=== ALL STUDY A RUNS COMPLETE ==="
