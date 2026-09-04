@echo off
REM One-shot clean benchmark: self-play then GPU sampling.
REM Atomic directory lock: only the first instance proceeds.
cd /d %~dp0
mkdir bench.lock 2>nul || exit /b 0
python benchmarks\bench_selfplay.py --config configs/combined.yaml --minutes 1 > bench_sp_after.log 2>&1
python benchmarks\gpu_monitor.py 60 gpu_util_engine2.csv
rmdir bench.lock