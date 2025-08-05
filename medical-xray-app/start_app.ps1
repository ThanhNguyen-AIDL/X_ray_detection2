# Medical X-Ray Application Startup Script
Write-Host "Starting Medical X-Ray Application..." -ForegroundColor Green
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

try {
    # Start backend server
    Write-Host "Starting backend server..." -ForegroundColor Yellow
    $backendDir = Join-Path $scriptDir "backend"
    $venvActivate = "D:\AITest\AIXRAY\yolov5-venv\Scripts\Activate.ps1"
    
    # Start backend in new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; & '$venvActivate'; python simple_api.py" -WindowStyle Normal
    
    # Wait for backend to start
    Write-Host "Waiting for backend to initialize..." -ForegroundColor Cyan
    Start-Sleep -Seconds 5
    
    # Start frontend server
    Write-Host "Starting frontend development server..." -ForegroundColor Yellow
    $frontendDir = Join-Path $scriptDir "frontend"
    
    # Start frontend in new PowerShell window
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; npm run dev" -WindowStyle Normal
    
    Write-Host ""
    Write-Host "Both servers are starting up..." -ForegroundColor Green
    Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    
} catch {
    Write-Host "Error starting application: $_" -ForegroundColor Red
}

# Wait for user input
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
