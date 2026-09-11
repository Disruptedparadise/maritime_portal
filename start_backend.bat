@echo off
cd /d %~dp0
echo.
echo  =========================================
echo   GreenFleet Optimizer - API Backend
echo   http://localhost:8000
echo   http://localhost:8000/portal  (Maritime Portal)
echo   http://localhost:8000/docs    (API Docs)
echo  =========================================
echo.
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
pause
