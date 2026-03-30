param(
    [string]$RustToolchain = '1.90.0-x86_64-pc-windows-msvc',
    [string]$VcpkgRoot = 'C:\vcpkg',
    [string]$RustRoot = 'C:\Rust',
    [string]$BuildToolsPath = 'C:\BuildTools',
    [string]$Triplet = 'x64-windows'
)

$ErrorActionPreference = 'Stop'

function Add-MachinePathEntry {
    param([string]$Entry)

    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $segments = @()
    if ($machinePath) {
        $segments = $machinePath.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
    }
    if ($segments -notcontains $Entry) {
        $newPath = if ($machinePath) { "$machinePath;$Entry" } else { $Entry }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
    }
}

function Normalize-PathString {
    param([string]$Path)

    if (-not $Path) {
        return $null
    }

    return [System.IO.Path]::GetFullPath($Path.Trim()).TrimEnd('\')
}

function Require-Admin {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script from an elevated PowerShell session.'
    }
}

function Assert-PathMissing {
    param(
        [string]$Path,
        [string]$Description
    )

    if (Test-Path $Path) {
        throw "$Description already exists at $Path. This bootstrap expects a clean machine and does not remove previous installations."
    }
}

function Assert-CleanWindowsBootstrapState {
    param(
        [string]$RustRoot,
        [string]$VcpkgRoot,
        [string]$BuildToolsPath
    )

    $userProfile = [Environment]::GetFolderPath('UserProfile')
    foreach ($path in @(
        @{ Path = (Join-Path $userProfile '.cargo'); Description = 'A per-user Rust cargo directory' },
        @{ Path = (Join-Path $userProfile '.rustup'); Description = 'A per-user Rust rustup directory' },
        @{ Path = $RustRoot; Description = 'The global Rust root' },
        @{ Path = $VcpkgRoot; Description = 'The global vcpkg root' },
        @{ Path = $BuildToolsPath; Description = 'The Visual Studio Build Tools install path' }
    )) {
        Assert-PathMissing -Path $path.Path -Description $path.Description
    }
}

Require-Admin

$toolsRoot = 'C:\Tools'
$cargoHome = Join-Path $RustRoot '.cargo'
$rustupHome = Join-Path $RustRoot '.rustup'
$cargoBin = Join-Path $cargoHome 'bin'
$vcpkgInstalled = Join-Path $VcpkgRoot "installed\$Triplet"
$perlBin = 'C:\Strawberry\perl\bin'
$perlExe = Join-Path $perlBin 'perl.exe'
$expectedRustup = Normalize-PathString (Join-Path $cargoBin 'rustup.exe')
$expectedCargo = Normalize-PathString (Join-Path $cargoBin 'cargo.exe')

Assert-CleanWindowsBootstrapState -RustRoot $RustRoot -VcpkgRoot $VcpkgRoot -BuildToolsPath $BuildToolsPath

New-Item -ItemType Directory -Force $toolsRoot | Out-Null
New-Item -ItemType Directory -Force $cargoHome | Out-Null
New-Item -ItemType Directory -Force $rustupHome | Out-Null

Write-Host 'Installing Visual Studio 2022 Build Tools with MSVC support'
$vsInstaller = Join-Path $toolsRoot 'vs_BuildTools.exe'
Invoke-WebRequest 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $vsInstaller
Start-Process -FilePath $vsInstaller -ArgumentList @(
    '--quiet',
    '--wait',
    '--norestart',
    '--nocache',
    '--installPath', $BuildToolsPath,
    '--add', 'Microsoft.VisualStudio.Workload.VCTools',
    '--includeRecommended'
) -Wait -NoNewWindow

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $vsInstall) {
    throw 'Visual Studio C++ Build Tools were not detected after installation.'
}
Write-Host "Detected Visual Studio Build Tools at: $vsInstall"

Write-Host 'Installing helper tools'
Set-ExecutionPolicy Bypass -Scope Process -Force
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}
choco install -y git pkgconfiglite 7zip strawberryperl

Write-Host 'Installing Rust toolchain'
$env:CARGO_HOME = $cargoHome
$env:RUSTUP_HOME = $rustupHome
[Environment]::SetEnvironmentVariable('CARGO_HOME', $cargoHome, 'Machine')
[Environment]::SetEnvironmentVariable('RUSTUP_HOME', $rustupHome, 'Machine')
$env:Path = "$cargoBin;$VcpkgRoot;$perlBin;C:\Program Files\Git\cmd;$env:Path"
$rustupInstaller = Join-Path $toolsRoot 'rustup-init.exe'
Invoke-WebRequest 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe' -OutFile $rustupInstaller
& $rustupInstaller @(
    '-y',
    '--default-host', 'x86_64-pc-windows-msvc',
    '--default-toolchain', $RustToolchain,
    '--profile', 'minimal'
) 
if ($LASTEXITCODE -ne 0) {
    throw "rustup-init failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path (Join-Path $cargoBin 'rustup.exe'))) {
    throw "rustup.exe was not installed to $(Join-Path $cargoBin 'rustup.exe')."
}
& (Join-Path $cargoBin 'rustup.exe') component add --toolchain $RustToolchain rustfmt clippy

Write-Host 'Cloning and bootstrapping vcpkg'
if (-not (Test-Path $VcpkgRoot)) {
    git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
}
& (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics

Write-Host 'Installing vcpkg packages used by Windows CI'
& (Join-Path $VcpkgRoot 'vcpkg.exe') install "openblas:$Triplet"
& (Join-Path $VcpkgRoot 'vcpkg.exe') install "lapack-reference:$Triplet"
& (Join-Path $VcpkgRoot 'vcpkg.exe') install "zeromq:$Triplet"

Write-Host 'Writing machine-wide environment variables'
[Environment]::SetEnvironmentVariable('CARGO_HOME', $cargoHome, 'Machine')
[Environment]::SetEnvironmentVariable('RUSTUP_HOME', $rustupHome, 'Machine')
[Environment]::SetEnvironmentVariable('VCPKG_ROOT', $VcpkgRoot, 'Machine')
[Environment]::SetEnvironmentVariable('VCPKG_DEFAULT_TRIPLET', $Triplet, 'Machine')
[Environment]::SetEnvironmentVariable('VCPKGRS_TRIPLET', $Triplet, 'Machine')
[Environment]::SetEnvironmentVariable('VCPKGRS_DYNAMIC', '1', 'Machine')
[Environment]::SetEnvironmentVariable('OPENBLAS_DIR', $vcpkgInstalled, 'Machine')
[Environment]::SetEnvironmentVariable('BLAS_LIB_DIR', (Join-Path $vcpkgInstalled 'lib'), 'Machine')
[Environment]::SetEnvironmentVariable('BLAS_LIBS', 'openblas', 'Machine')
[Environment]::SetEnvironmentVariable('LAPACK_LIB_DIR', (Join-Path $vcpkgInstalled 'lib'), 'Machine')
[Environment]::SetEnvironmentVariable('LAPACK_LIBS', 'lapack;openblas', 'Machine')
[Environment]::SetEnvironmentVariable('ZMQ_PATH', $vcpkgInstalled, 'Machine')
[Environment]::SetEnvironmentVariable('ZMQ_INCLUDE_DIR', (Join-Path $vcpkgInstalled 'include'), 'Machine')
[Environment]::SetEnvironmentVariable('ZMQ_LIB_DIR', (Join-Path $vcpkgInstalled 'lib'), 'Machine')
[Environment]::SetEnvironmentVariable('PKG_CONFIG_PATH', (Join-Path $vcpkgInstalled 'lib\pkgconfig'), 'Machine')
[Environment]::SetEnvironmentVariable('PERL', $perlExe, 'Machine')

Add-MachinePathEntry $cargoBin
Add-MachinePathEntry $VcpkgRoot
Add-MachinePathEntry (Join-Path $vcpkgInstalled 'bin')
Add-MachinePathEntry 'C:\Program Files\Git\cmd'
Add-MachinePathEntry $perlBin

Write-Host 'Validating installed toolchain'
& (Join-Path $cargoBin 'rustc.exe') -V
& (Join-Path $cargoBin 'cargo.exe') -V
& (Join-Path $cargoBin 'rustup.exe') --version
& (Join-Path $VcpkgRoot 'vcpkg.exe') version
if (-not (Test-Path $perlExe)) {
    throw "Perl executable was not found at $perlExe after installation."
}
if (-not (Get-Command perl -ErrorAction SilentlyContinue)) {
    throw 'Perl was not found on PATH after installation.'
}
$rustupLocations = @(where.exe rustup 2>$null) | Where-Object { $_ }
$normalizedRustupLocations = @($rustupLocations | ForEach-Object { Normalize-PathString $_ }) | Where-Object { $_ }
if ($normalizedRustupLocations.Count -eq 0 -or $normalizedRustupLocations[0] -ne $expectedRustup) {
    throw "rustup resolution is not deterministic. Expected first rustup on PATH to be $expectedRustup, got: $($normalizedRustupLocations -join ', ')"
}
$cargoLocations = @(where.exe cargo 2>$null) | Where-Object { $_ }
$normalizedCargoLocations = @($cargoLocations | ForEach-Object { Normalize-PathString $_ }) | Where-Object { $_ }
if ($normalizedCargoLocations.Count -eq 0 -or $normalizedCargoLocations[0] -ne $expectedCargo) {
    throw "cargo resolution is not deterministic. Expected first cargo on PATH to be $expectedCargo, got: $($normalizedCargoLocations -join ', ')"
}
& (Join-Path $cargoBin 'rustup.exe') which --toolchain $RustToolchain cargo | Out-Null

$requiredPaths = @(
    (Join-Path $vcpkgInstalled 'include\zmq.h'),
    (Join-Path $vcpkgInstalled 'lib\openblas.lib'),
    (Join-Path $vcpkgInstalled 'lib\lapack.lib'),
    (Join-Path $vcpkgInstalled 'lib\pkgconfig\libzmq.pc')
)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        throw "Missing expected dependency artifact: $path"
    }
}

$zmqImportLibs = Get-ChildItem -Path (Join-Path $vcpkgInstalled 'lib') -Filter '*zmq*.lib' -ErrorAction SilentlyContinue
if (-not $zmqImportLibs) {
    throw "Missing expected ZeroMQ import library under $(Join-Path $vcpkgInstalled 'lib')"
}

Write-Host ''
Write-Host 'Windows runner bootstrap complete.'
Write-Host 'Restart the GitHub Actions runner service before rerunning CI.'
