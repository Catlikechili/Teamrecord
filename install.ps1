# Cai dat tu dong Verbatim Scientific Recorder
# Cac buoc: kiem tra/cai Python -> tao moi truong ao .venv -> cai thu vien -> tai truoc mo hinh Whisper -> tao shortcut.
# Tham so: -NoShortcut (khong tao shortcut Desktop), -SkipModel (khong tai truoc mo hinh)
param(
    [switch]$NoShortcut,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host "[LOI] $msg" -ForegroundColor Red; exit 1 }

function Find-Python {
    # Tra ve lenh goi Python >= 3.10, hoac $null
    $candidates = @(
        @("py", "-3.12"), @("py", "-3.13"), @("py", "-3.11"), @("py", "-3.10"), @("py", "-3"),
        @("python"), @("python3")
    )
    foreach ($c in $candidates) {
        try {
            $exe = $c[0]
            $args_ = @()
            if ($c.Length -gt 1) { $args_ = $c[1..($c.Length - 1)] }
            $ver = & $exe @args_ -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $parts = $ver.Trim().Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) { return , $c }
            }
        } catch { }
    }
    return $null
}

Write-Host "=================================================================="
Write-Host "  VERBATIM SCIENTIFIC RECORDER - CAI DAT TU DONG"
Write-Host "  Thu muc: $Root"
Write-Host "=================================================================="

# 1. Python
Step "Kiem tra Python 3.10+"
$py = Find-Python
if (-not $py) {
    Write-Host "Chua co Python 3.10+. Thu cai Python 3.12 bang winget..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        $py = Find-Python
    }
    if (-not $py) {
        Start-Process "https://www.python.org/downloads/windows/"
        Fail "Khong tu cai duoc Python. Hay cai Python 3.12 (tick 'Add python.exe to PATH') roi chay lai install.bat."
    }
}
$pyExe = $py[0]
$pyArgs = @()
if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }
Write-Host ("Dung: " + ($py -join " ") + "  (" + (& $pyExe @pyArgs --version) + ")")

# 2. Moi truong ao
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPy)) {
    Step "Tao moi truong ao .venv"
    & $pyExe @pyArgs -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "Khong tao duoc moi truong ao." }
} else {
    Step "Da co moi truong ao .venv - dung lai"
}

# 3. Thu vien
Step "Cai thu vien (lan dau mat vai phut)"
& $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
if ($LASTEXITCODE -ne 0) { Fail "Khong nang cap duoc pip (kiem tra ket noi mang)." }
& $venvPy -m pip install -r (Join-Path $Root "requirements.txt") --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "Cai thu vien that bai. Xem thong bao loi ben tren." }

# 4. Mo hinh Whisper mac dinh (base, ~145 MB) de lan chay dau khong phai cho
if (-not $SkipModel) {
    Step "Tai truoc mo hinh Whisper 'base' (~145 MB)"
    & $venvPy -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('OK')"
    if ($LASTEXITCODE -ne 0) { Write-Host "Chua tai duoc mo hinh - ung dung se tu tai o lan chay dau." -ForegroundColor Yellow }
}

# 5. Kiem tra nhanh
Step "Kiem tra thu vien va thiet bi am thanh"
& $venvPy -c "import core_engine as c; d=c.AudioLoopbackCapture.list_loopback_devices(); print('Thiet bi loopback:', len(d)); [print('  -', x['name']) for x in d]"
if ($LASTEXITCODE -ne 0) { Fail "Nap ung dung that bai." }

# 6. Shortcut Desktop
if (-not $NoShortcut) {
    Step "Tao shortcut tren Desktop"
    $desktop = [Environment]::GetFolderPath("Desktop")
    $lnk = Join-Path $desktop "Verbatim Recorder.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut($lnk)
    $sc.TargetPath = Join-Path $Root "run.bat"
    $sc.WorkingDirectory = $Root
    $sc.WindowStyle = 7
    $sc.Description = "Ghi nguyen van loi noi cuoc hop"
    $sc.Save()
    Write-Host "Da tao: $lnk"
}

Write-Host ""
Write-Host "HOAN TAT. Mo ung dung bang shortcut 'Verbatim Recorder' tren Desktop hoac file run.bat." -ForegroundColor Green
