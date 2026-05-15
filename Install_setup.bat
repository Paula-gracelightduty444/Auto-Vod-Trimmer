@echo off
color 0A
echo ===================================================
echo   VOD Auto Trimmer - First Time Setup Installer
echo ===================================================
echo.

:: Step 1: Verify Python is installed and in PATH
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python 3.12, and MAKE SURE to check the box
    echo that says "Add Python to PATH" at the bottom of the installer.
    echo.
    pause
    exit /b
)
echo [*] Python detected successfully.

:: Step 2: Install Python Libraries
echo.
echo [*] Installing required AI and Audio Python libraries...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Step 3: Check and Install FFmpeg via Winget
echo.
echo [*] Checking for FFmpeg (Required for video processing)...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    color 0E
    echo [!] FFmpeg not found! Attempting to install via Windows Package Manager...
    
    :: Using BtbN's build which natively uses Winget's symlink alias system
    winget install -e --id BtbN.FFmpeg.GPL --accept-source-agreements --accept-package-agreements
    
    echo.
    echo [*] Syncing Environment Variables...
    :: This trick dynamically reloads the PATH from the Windows Registry so no terminal restart is needed
    for /f "tokens=2* delims= " %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
    for /f "tokens=2* delims= " %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
    set "PATH=%USER_PATH%;%SYS_PATH%;%PATH%"
    
    echo [*] FFmpeg linked successfully!
) else (
    echo [*] FFmpeg is already installed and ready.
)

echo.
color 0A
echo ===================================================
echo   SETUP COMPLETE! 
echo   You can now double-click "run_trimmer.bat"
echo ===================================================
pause