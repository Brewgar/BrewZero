@echo off
cd /d %~dp0
mkdir test.lock 2>nul || exit /b 0
.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider > test_run.log 2>&1
rmdir test.lock