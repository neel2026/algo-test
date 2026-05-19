@echo off
start "" "%~dp0start_backend.bat"
timeout /t 3 /nobreak >nul
start "" "%~dp0start_frontend.bat"
timeout /t 2 /nobreak >nul
start http://localhost:5173
