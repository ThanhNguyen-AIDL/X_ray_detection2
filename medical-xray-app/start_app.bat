@echo off
echo Starting Medical X-Ray Application...
echo.

REM Start backend in background
echo Starting backend server...
cd /d "%~dp0backend"
start "Backend Server" cmd /c "call D:\AITest\AIXRAY\yolov5-venv\Scripts\activate.bat && python simple_api.py"

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend
echo Starting frontend development server...
cd /d "%~dp0frontend"
start "Frontend Server" cmd /c "npm run dev"

echo.
echo Both servers are starting up...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause >nul
