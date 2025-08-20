# RunMat Installation Script for Windows
# Usage: iwr https://runmat.org/install.ps1 | iex

$ErrorActionPreference = "Stop"

# Constants
$REPO = "runmat-org/runmat"
$BINARY_NAME = "runmat.exe"
$INSTALL_DIR = "$env:USERPROFILE\.runmat\bin"
$WEBSITE_URL = "https://runmat.org"

$TELEMETRY_ENDPOINT = "https://runmat.org/api/telemetry"
$TELEMETRY_ID_FILE = "$env:USERPROFILE\.runmat\telemetry_id"

function New-AnonymousClientId {
    try {
        if (Test-Path $TELEMETRY_ID_FILE) {
            return Get-Content $TELEMETRY_ID_FILE -ErrorAction SilentlyContinue
        }
        $cid = [guid]::NewGuid().ToString()
        $dir = Split-Path $TELEMETRY_ID_FILE
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Set-Content -Path $TELEMETRY_ID_FILE -Value $cid -ErrorAction SilentlyContinue
        return $cid
    } catch {
        return "anon-" + (Get-Random)
    }
}

function Send-Telemetry {
    param(
        [Parameter(Mandatory=$true)][string]$EventName,
        [string]$OS,
        [string]$ARCH,
        [string]$PLATFORM,
        [string]$Release
    )
    # Honor opt-out via environment variables
    if ($env:RUNMAT_NO_TELEMETRY -or $env:RUNMAT_TELEMETRY -eq '0') { return }
    try {
        $cid = New-AnonymousClientId
        $payload = @{ 
            event_label = $EventName
            os = $OS
            arch = $ARCH
            platform = $PLATFORM
            release = $Release
            method = "powershell"
            cid = $cid
        } | ConvertTo-Json -Compress
        Invoke-WebRequest -Method Post -Uri $TELEMETRY_ENDPOINT -Body $payload -ContentType "application/json" -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null
    } catch {}
}

# Functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param($Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

# Banner
Write-Host @"
  _____                 __  __       _   
 |  __ \               |  \/  |     | |  
 | |__) | _   _ _ __   | \  / | __ _| |_ 
 |  _  / | | | | '_ \  | |\/| |/ _` | __|
 | | \ \ |_| | | | | | | |  | | (_| | |_ 
 |_|  \_\__,_|_| |_| |_|_|  |_|\__,_|\__|

High-performance MATLAB/Octave runtime
"@ -ForegroundColor Blue

Write-Info "Starting RunMat installation..."

# Detect architecture
$ARCH = $env:PROCESSOR_ARCHITECTURE
Write-Info "Detected architecture: $ARCH"

switch ($ARCH) {
    "AMD64" { $PLATFORM = "windows-x86_64" }
    "ARM64" { $PLATFORM = "windows-aarch64" }
    default { 
        Write-ErrorMsg "Unsupported architecture: $ARCH"
    }
}

Write-Info "Installing for platform: $PLATFORM"

# Send start event
try { Send-Telemetry -EventName "install_start" -OS "windows" -ARCH $ARCH -PLATFORM $PLATFORM -Release "unknown" } catch {}

# Check PowerShell version
if ($PSVersionTable.PSVersion.Major -lt 3) {
    Write-ErrorMsg "PowerShell 3.0 or later is required"
}

# Get latest release
Write-Info "Fetching latest release information..."
try {
    $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$REPO/releases/latest"
    $LATEST_RELEASE = $response.tag_name
} catch {
    Write-ErrorMsg "Failed to get latest release information: $($_.Exception.Message)"
}

if (-not $LATEST_RELEASE) {
    Write-ErrorMsg "Failed to get latest release information"
}

Write-Info "Latest release: $LATEST_RELEASE"

# Determine which artifact to download
$baseUrl = "https://github.com/$REPO/releases/download/$LATEST_RELEASE"
$DownloadPlatform = $PLATFORM
if ($PLATFORM -eq "windows-aarch64") {
    Write-Warn "No native Windows ARM64 build is published yet; using the x64 build via Windows on ARM emulation."
    $DownloadPlatform = "windows-x86_64"
}

# Download and install (try versioned and non-versioned names)
$candidates = @(
  "$baseUrl/runmat-$LATEST_RELEASE-$DownloadPlatform.zip",
  "$baseUrl/runmat-$DownloadPlatform.zip"
)
$TEMP_FILE = "$env:TEMP\runmat-$DownloadPlatform.zip"
$TEMP_DIR = "$env:TEMP\runmat-extract"

$downloaded = $false
foreach ($url in $candidates) {
    Write-Info "Downloading from: $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $TEMP_FILE -ErrorAction Stop
        $downloaded = $true
        break
    } catch {
        Write-Warn "Download failed from $($url): $($_.Exception.Message)"
    }
}
if (-not $downloaded) { Write-ErrorMsg "Failed to download RunMat; tried: $($candidates -join ', ')" }

# Extract
Write-Info "Extracting RunMat..."
if (Test-Path $TEMP_DIR) {
    Remove-Item $TEMP_DIR -Recurse -Force
}

try {
    Expand-Archive -Path $TEMP_FILE -DestinationPath $TEMP_DIR
} catch {
    Write-ErrorMsg "Failed to extract RunMat: $($_.Exception.Message)"
}

# Create install directory
Write-Info "Creating installation directory: $INSTALL_DIR"
if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
}

# Install binary and any bundled DLLs (do not overwrite system-wide deps)
Write-Info "Installing RunMat binary..."
try {
    Copy-Item "$TEMP_DIR\$BINARY_NAME" "$INSTALL_DIR\" -Force
    # Copy any DLLs that shipped alongside the binary into the private install dir
    Get-ChildItem -Path $TEMP_DIR -Filter *.dll -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName "$INSTALL_DIR\" -Force
    }
} catch {
    Write-ErrorMsg "Failed to install binary: $($_.Exception.Message)"
}

# Cleanup
Remove-Item $TEMP_FILE -Force -ErrorAction SilentlyContinue
Remove-Item $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue

Write-Info "RunMat installed successfully to $INSTALL_DIR\$BINARY_NAME"

# Add to PATH for user and current session if not already present
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$INSTALL_DIR*") {
    try {
        Write-Info "Adding $INSTALL_DIR to your PATH (User scope)..."
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $INSTALL_DIR } else { "$INSTALL_DIR;$userPath" }
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Info "Added $INSTALL_DIR to your PATH (User)"
    } catch {
        Write-Warn "Could not set PATH in User scope: $($_.Exception.Message)"
    }
}

# Add to current process PATH to be immediately available
if ($env:PATH -notlike "*$INSTALL_DIR*") {
    $env:PATH = "$INSTALL_DIR;$env:PATH"
}

Write-Warn "If other terminals were open, restart them to pick up PATH changes."

# Test installation
try {
    if (Get-Command runmat -ErrorAction SilentlyContinue) {
        $INSTALLED_VERSION = & runmat --version 2>$null
        Write-Info "Installation verified! Version: $INSTALLED_VERSION"
    } elseif (Test-Path "$INSTALL_DIR\runmat.exe") {
        $INSTALLED_VERSION = & "$INSTALL_DIR\runmat.exe" --version 2>$null
        Write-Info "Installation verified! Version: $INSTALLED_VERSION"
    } else {
        Write-ErrorMsg "Installation verification failed"
    }
} catch {
    Write-Warn "Could not verify installation, but binary is installed at $INSTALL_DIR\$BINARY_NAME"
}

# Completion event
try { Send-Telemetry -EventName "install_complete" -OS "windows" -ARCH $ARCH -PLATFORM $PLATFORM -Release $LATEST_RELEASE } catch {}

Write-Host ""
Write-Info "Installation complete! 🎉"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start the interactive REPL: runmat"
Write-Host "  2. Run a script: runmat run script.m"
Write-Host "  3. Install Jupyter kernel: runmat --install-kernel"
Write-Host "  4. Get help: runmat --help"
Write-Host ""
Write-Host "Documentation: $WEBSITE_URL/docs" -ForegroundColor Blue
Write-Host "Getting Started: $WEBSITE_URL/docs/getting-started" -ForegroundColor Blue
Write-Host ""
Write-Info "Happy computing with RunMat!"