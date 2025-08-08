# RustMat Installation Script for Windows
# Usage: iwr https://rustmat.com/install.ps1 | iex

$ErrorActionPreference = "Stop"

# Constants
$REPO = "rustmat/rustmat"
$BINARY_NAME = "rustmat.exe"
$INSTALL_DIR = "$env:USERPROFILE\.rustmat\bin"
$WEBSITE_URL = "https://rustmat.com"

# Functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param($Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

# Banner
Write-Host @"
 ____            _   __  __       _   
|  _ \ _   _ ___| |_|  \/  | __ _| |_ 
| |_) | | | / __| __| |\/| |/ _` | __|
|  _ <| |_| \__ \ |_| |  | | (_| | |_ 
|_| \_\__,_|___/\__|_|  |_|\__,_|\__|

High-performance MATLAB/Octave runtime
"@ -ForegroundColor Blue

Write-Info "Starting RustMat installation..."

# Detect architecture
$ARCH = $env:PROCESSOR_ARCHITECTURE
Write-Info "Detected architecture: $ARCH"

switch ($ARCH) {
    "AMD64" { $PLATFORM = "Windows-x86_64" }
    "ARM64" { $PLATFORM = "Windows-aarch64" }
    default { 
        Write-Error "Unsupported architecture: $ARCH"
    }
}

Write-Info "Installing for platform: $PLATFORM"

# Check PowerShell version
if ($PSVersionTable.PSVersion.Major -lt 3) {
    Write-Error "PowerShell 3.0 or later is required"
}

# Get latest release
Write-Info "Fetching latest release information..."
try {
    $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$REPO/releases/latest"
    $LATEST_RELEASE = $response.tag_name
} catch {
    Write-Error "Failed to get latest release information: $($_.Exception.Message)"
}

if (-not $LATEST_RELEASE) {
    Write-Error "Failed to get latest release information"
}

Write-Info "Latest release: $LATEST_RELEASE"

# Download and install
$DOWNLOAD_URL = "https://github.com/$REPO/releases/download/$LATEST_RELEASE/rustmat-$PLATFORM.zip"
$TEMP_FILE = "$env:TEMP\rustmat-$PLATFORM.zip"
$TEMP_DIR = "$env:TEMP\rustmat-extract"

Write-Info "Downloading from: $DOWNLOAD_URL"
try {
    Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile $TEMP_FILE
} catch {
    Write-Error "Failed to download RustMat: $($_.Exception.Message)"
}

# Extract
Write-Info "Extracting RustMat..."
if (Test-Path $TEMP_DIR) {
    Remove-Item $TEMP_DIR -Recurse -Force
}

try {
    Expand-Archive -Path $TEMP_FILE -DestinationPath $TEMP_DIR
} catch {
    Write-Error "Failed to extract RustMat: $($_.Exception.Message)"
}

# Create install directory
Write-Info "Creating installation directory: $INSTALL_DIR"
if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
}

# Install binary
Write-Info "Installing RustMat binary..."
try {
    Copy-Item "$TEMP_DIR\$BINARY_NAME" "$INSTALL_DIR\" -Force
} catch {
    Write-Error "Failed to install binary: $($_.Exception.Message)"
}

# Cleanup
Remove-Item $TEMP_FILE -Force -ErrorAction SilentlyContinue
Remove-Item $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue

Write-Info "RustMat installed successfully to $INSTALL_DIR\$BINARY_NAME"

# Add to PATH if not already there
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$INSTALL_DIR*") {
    Write-Info "Adding $INSTALL_DIR to your PATH..."
    $newPath = "$INSTALL_DIR;$userPath"
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Info "Added $INSTALL_DIR to your PATH"
    Write-Warn "Please restart your terminal to use rustmat from anywhere"
    
    # Also add to current session
    $env:PATH = "$INSTALL_DIR;$env:PATH"
} else {
    Write-Info "RustMat is already in your PATH"
}

# Test installation
try {
    if (Get-Command rustmat -ErrorAction SilentlyContinue) {
        $INSTALLED_VERSION = & rustmat --version 2>$null
        Write-Info "Installation verified! Version: $INSTALLED_VERSION"
    } elseif (Test-Path "$INSTALL_DIR\rustmat.exe") {
        $INSTALLED_VERSION = & "$INSTALL_DIR\rustmat.exe" --version 2>$null
        Write-Info "Installation verified! Version: $INSTALLED_VERSION"
    } else {
        Write-Error "Installation verification failed"
    }
} catch {
    Write-Warn "Could not verify installation, but binary is installed at $INSTALL_DIR\$BINARY_NAME"
}

Write-Host ""
Write-Info "Installation complete! 🎉"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start the interactive REPL: rustmat"
Write-Host "  2. Run a script: rustmat run script.m"
Write-Host "  3. Install Jupyter kernel: rustmat --install-kernel"
Write-Host "  4. Get help: rustmat --help"
Write-Host ""
Write-Host "Documentation: $WEBSITE_URL/docs" -ForegroundColor Blue
Write-Host "Examples: $WEBSITE_URL/docs/examples" -ForegroundColor Blue
Write-Host ""
Write-Info "Happy computing with RustMat!"