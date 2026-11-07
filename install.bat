@echo off
rem Cai dat tu dong Verbatim Scientific Recorder (bam dup chuot de chay)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
