@echo off
rem Mo Verbatim Scientific Recorder (uu tien moi truong ao .venv do install.bat tao ra)
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" goto venv
where pythonw >nul 2>nul
if errorlevel 1 goto missing
start "" pythonw main.py
goto :eof

:venv
start "" ".venv\Scripts\pythonw.exe" main.py
goto :eof

:missing
echo Chua cai dat. Hay chay install.bat truoc.
pause
