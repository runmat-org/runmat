param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Triplet = $env:VCPKG_DEFAULT_TRIPLET,
    [string]$OverlayPorts = ''
)

$ErrorActionPreference = 'Stop'

if (-not $VcpkgRoot) {
    $VcpkgRoot = 'C:\vcpkg'
}
if (-not $Triplet) {
    $Triplet = 'x64-windows'
}
if (-not $OverlayPorts) {
    $repositoryRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
    $OverlayPorts = Join-Path $repositoryRoot 'infra\vcpkg-ports'
}

$vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
$prefix = Join-Path $VcpkgRoot "installed\$Triplet"
$libaecOverlay = Join-Path $OverlayPorts 'libaec\vcpkg.json'

if (-not (Test-Path -LiteralPath $vcpkg)) {
    throw "vcpkg.exe was not found at $vcpkg"
}
if (-not (Test-Path -LiteralPath $libaecOverlay)) {
    throw "RunMat's libaec overlay was not found at $libaecOverlay"
}

function Install-VcpkgPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Package,
        [string[]]$ExtraArguments = @(),
        [int]$MaximumAttempts = 3
    )

    for ($attempt = 1; $attempt -le $MaximumAttempts; $attempt++) {
        & $vcpkg install "${Package}:$Triplet" "--overlay-ports=$OverlayPorts" @ExtraArguments
        if ($LASTEXITCODE -eq 0) {
            return
        }

        if ($attempt -eq $MaximumAttempts) {
            throw "vcpkg install ${Package}:$Triplet failed after $MaximumAttempts attempts"
        }

        $delaySeconds = 15 * [Math]::Pow(2, $attempt - 1)
        Write-Warning "vcpkg install ${Package}:$Triplet failed on attempt $attempt of $MaximumAttempts; retrying in $delaySeconds seconds"
        Start-Sleep -Seconds $delaySeconds
    }
}

$opencascadeExtraArguments = @()
if ($Triplet -eq 'arm64-windows') {
    $opencascadeExtraArguments = @('--allow-unsupported')
}

$packageSpecifications = @(
    [pscustomobject]@{
        Name = 'openblas'
        RequiredArtifacts = @('lib\openblas.lib')
        ExtraArguments = @()
    },
    [pscustomobject]@{
        Name = 'clapack'
        RequiredArtifacts = @('lib\lapack.lib')
        ExtraArguments = @()
    },
    [pscustomobject]@{
        Name = 'zeromq'
        RequiredArtifacts = @('include\zmq.h', 'lib\pkgconfig\libzmq.pc')
        ExtraArguments = @()
    },
    [pscustomobject]@{
        Name = 'hdf5'
        RequiredArtifacts = @('include\H5pubconf.h', 'lib\hdf5.lib', 'bin\hdf5.dll')
        ExtraArguments = @()
    },
    [pscustomobject]@{
        Name = 'opencascade'
        RequiredArtifacts = @('include\opencascade', 'lib\TKBRep.lib')
        ExtraArguments = $opencascadeExtraArguments
    }
)

foreach ($specification in $packageSpecifications) {
    $missingArtifacts = @(
        foreach ($relativePath in $specification.RequiredArtifacts) {
            $path = Join-Path $prefix $relativePath
            if (-not (Test-Path -LiteralPath $path)) {
                $path
            }
        }
    )

    if ($missingArtifacts.Count -eq 0) {
        Write-Host "$($specification.Name): required artifacts are already installed"
        continue
    }

    Write-Host "$($specification.Name): installing because these artifacts are missing:"
    $missingArtifacts | ForEach-Object { Write-Host "  $_" }
    Install-VcpkgPackage `
        -Package $specification.Name `
        -ExtraArguments $specification.ExtraArguments
}

$requiredArtifacts = @(
    'include\zmq.h',
    'include\H5pubconf.h',
    'include\opencascade',
    'lib\openblas.lib',
    'lib\lapack.lib',
    'lib\hdf5.lib',
    'bin\hdf5.dll',
    'lib\TKBRep.lib',
    'lib\pkgconfig\libzmq.pc'
)

foreach ($relativePath in $requiredArtifacts) {
    $path = Join-Path $prefix $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required Windows dependency artifact is missing after vcpkg maintenance: $path"
    }
}

$zmqImportLibraries = @(
    Get-ChildItem -LiteralPath (Join-Path $prefix 'lib') -Filter '*zmq*.lib' -File -ErrorAction SilentlyContinue
)
if ($zmqImportLibraries.Count -eq 0) {
    throw "Required ZeroMQ import library is missing under $(Join-Path $prefix 'lib')"
}

Write-Host "Windows vcpkg dependencies are ready under $prefix"
