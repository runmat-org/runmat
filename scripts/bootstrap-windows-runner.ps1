param(
    [string]$RustToolchain = '1.90.0-x86_64-pc-windows-msvc',
    [string]$VcpkgRoot = 'C:\vcpkg',
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

function Get-ResolvedCommandPath {
    param([string]$Name)

    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) {
        return $null
    }

    return Normalize-PathString $command.Source
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

function Assert-EnvironmentVariableUnset {
    param(
        [string]$Name,
        [string]$Scope
    )

    $value = [Environment]::GetEnvironmentVariable($Name, $Scope)
    if ($value) {
        throw "$Name is already defined for scope $Scope as '$value'. This bootstrap expects a clean machine and does not override prior Rust location settings."
    }
}

function Assert-CleanWindowsBootstrapState {
    param(
        [string]$RunnerUserProfile,
        [string]$VcpkgRoot,
        [string]$BuildToolsPath
    )

    foreach ($path in @(
        @{ Path = (Join-Path $RunnerUserProfile '.cargo'); Description = 'A per-user Rust cargo directory' },
        @{ Path = (Join-Path $RunnerUserProfile '.rustup'); Description = 'A per-user Rust rustup directory' },
        @{ Path = 'C:\Rust'; Description = 'A machine-global Rust root from an older provisioning model' },
        @{ Path = $VcpkgRoot; Description = 'The global vcpkg root' },
        @{ Path = $BuildToolsPath; Description = 'The Visual Studio Build Tools install path' }
    )) {
        Assert-PathMissing -Path $path.Path -Description $path.Description
    }

    foreach ($scope in @('Machine', 'User')) {
        Assert-EnvironmentVariableUnset -Name 'CARGO_HOME' -Scope $scope
        Assert-EnvironmentVariableUnset -Name 'RUSTUP_HOME' -Scope $scope
    }
}

Require-Admin

$toolsRoot = 'C:\Tools'
$runnerUserProfile = [Environment]::GetFolderPath('UserProfile')
$cargoHome = Join-Path $runnerUserProfile '.cargo'
$rustupHome = Join-Path $runnerUserProfile '.rustup'
$cargoBin = Join-Path $cargoHome 'bin'
$vcpkgInstalled = Join-Path $VcpkgRoot "installed\$Triplet"
$perlBin = 'C:\Strawberry\perl\bin'
$perlExe = Join-Path $perlBin 'perl.exe'
$expectedRustup = Normalize-PathString (Join-Path $cargoBin 'rustup.exe')
$expectedCargo = Normalize-PathString (Join-Path $cargoBin 'cargo.exe')

Assert-CleanWindowsBootstrapState -RunnerUserProfile $runnerUserProfile -VcpkgRoot $VcpkgRoot -BuildToolsPath $BuildToolsPath

New-Item -ItemType Directory -Force $toolsRoot | Out-Null
New-Item -ItemType Directory -Force $cargoHome | Out-Null
New-Item -ItemType Directory -Force $rustupHome | Out-Null

Write-Host 'Installing Visual Studio 2022 Build Tools with MSVC support'
$vsInstaller = Join-Path $toolsRoot 'vs_BuildTools.exe'
Invoke-WebRequest 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $vsInstaller
$vsProcess = Start-Process -FilePath $vsInstaller -ArgumentList @(
    '--quiet',
    '--wait',
    '--norestart',
    '--nocache',
    '--installPath', $BuildToolsPath,
    '--add', 'Microsoft.VisualStudio.Workload.VCTools',
    '--includeRecommended'
) -Wait -NoNewWindow -PassThru

if ($vsProcess.ExitCode -ne 0) {
    throw "Visual Studio Build Tools installer failed with exit code $($vsProcess.ExitCode). Check the Visual Studio installer logs under $env:TEMP and %ProgramData%\Microsoft\VisualStudio\Packages\_Instances."
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & $vswhere `
    -latest `
    -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath

if (-not $vsInstall) {
    $allVsInstalls = & $vswhere -all -products * -format json
    throw "Visual Studio C++ Build Tools were not detected after installation. vswhere output: $allVsInstalls"
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

Add-MachinePathEntry $VcpkgRoot
Add-MachinePathEntry (Join-Path $vcpkgInstalled 'bin')
Add-MachinePathEntry 'C:\Program Files\Git\cmd'
Add-MachinePathEntry $perlBin
$env:Path = "$perlBin;$VcpkgRoot;$(Join-Path $vcpkgInstalled 'bin');C:\Program Files\Git\cmd;$cargoBin;$env:Path"

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
$resolvedRustup = Get-ResolvedCommandPath 'rustup'
if (-not [string]::Equals($resolvedRustup, $expectedRustup, [System.StringComparison]::OrdinalIgnoreCase)) {
    $rustupLocations = @((where.exe rustup 2>$null) | Where-Object { $_ } | ForEach-Object { Normalize-PathString $_ }) | Where-Object { $_ }
    throw "rustup resolution is not deterministic. Expected first rustup on PATH to be $expectedRustup, got command resolution '$resolvedRustup' and where.exe entries: $($rustupLocations -join ', ')"
}
$resolvedCargo = Get-ResolvedCommandPath 'cargo'
if (-not [string]::Equals($resolvedCargo, $expectedCargo, [System.StringComparison]::OrdinalIgnoreCase)) {
    $cargoLocations = @((where.exe cargo 2>$null) | Where-Object { $_ } | ForEach-Object { Normalize-PathString $_ }) | Where-Object { $_ }
    throw "cargo resolution is not deterministic. Expected first cargo on PATH to be $expectedCargo, got command resolution '$resolvedCargo' and where.exe entries: $($cargoLocations -join ', ')"
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
Write-Host "Rust is installed for runner account $env:USERNAME at $cargoHome"
Write-Host 'Restart the GitHub Actions runner service before rerunning CI.'
