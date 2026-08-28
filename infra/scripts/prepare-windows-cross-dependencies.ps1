param(
    [Parameter(Mandatory = $true)]
    [string]$Target,
    [string]$VcpkgRoot = (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'vcpkg-runmat')
)

$ErrorActionPreference = 'Stop'
$vcpkgRevision = 'a7eda31dc16994fcaa8587982eb833a8695f1b6f'
$triplet = switch ($Target) {
    'x86_64-pc-windows-msvc' { 'x64-windows' }
    'aarch64-pc-windows-msvc' { 'arm64-windows' }
    default { throw "Unsupported Windows MSVC cross-build target: $Target" }
}
$prefix = Join-Path $VcpkgRoot "installed\$triplet"
$overlayPorts = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')) 'infra\vcpkg-ports'

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    throw 'Chocolatey is required to prepare Windows cross-build dependencies.'
}

& choco install pkgconfiglite strawberryperl cmake --yes --no-progress --installargs 'ADD_CMAKE_TO_PATH=System'
if ($LASTEXITCODE -ne 0) {
    throw "Chocolatey dependency installation failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Path $VcpkgRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $VcpkgRoot '.git'))) {
    & git -C $VcpkgRoot init
    if ($LASTEXITCODE -ne 0) { throw "git init for vcpkg failed with exit code $LASTEXITCODE" }
    & git -C $VcpkgRoot remote add origin https://github.com/microsoft/vcpkg.git
    if ($LASTEXITCODE -ne 0) { throw "adding the vcpkg origin failed with exit code $LASTEXITCODE" }
}

& git -C $VcpkgRoot fetch --depth 1 origin $vcpkgRevision
if ($LASTEXITCODE -ne 0) { throw "fetching vcpkg revision $vcpkgRevision failed with exit code $LASTEXITCODE" }
& git -C $VcpkgRoot checkout --force $vcpkgRevision
if ($LASTEXITCODE -ne 0) { throw "checking out vcpkg revision $vcpkgRevision failed with exit code $LASTEXITCODE" }

$resolvedRevision = (& git -C $VcpkgRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $resolvedRevision -ne $vcpkgRevision) {
    throw "Expected vcpkg revision $vcpkgRevision, found $resolvedRevision"
}

$vcpkg = Join-Path $VcpkgRoot 'vcpkg.exe'
if (-not (Test-Path -LiteralPath $vcpkg)) {
    & (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "bootstrap-vcpkg.bat failed with exit code $LASTEXITCODE" }
}

# The pinned vcpkg finder can select an ambient LLVM/MinGW Fortran compiler on
# hosted runners. OpenBLAS must instead use the compiler managed by its port.
$finder = Join-Path $VcpkgRoot 'scripts\cmake\vcpkg_find_fortran.cmake'
$content = Get-Content -LiteralPath $finder -Raw
$needle = '    include(CMakeDetermineFortranCompiler)'
$replacement = @'
    include(CMakeDetermineFortranCompiler)

    if(CMAKE_HOST_WIN32 AND "${VCPKG_CHAINLOAD_TOOLCHAIN_FILE}" STREQUAL "" AND
       ("${VCPKG_TARGET_ARCHITECTURE}" STREQUAL "x86" OR "${VCPKG_TARGET_ARCHITECTURE}" STREQUAL "x64"))
        if(CMAKE_Fortran_COMPILER)
            message(STATUS "Ignoring ambient Fortran compiler '${CMAKE_Fortran_COMPILER}' so vcpkg uses internal MinGW gfortran")
            unset(CMAKE_Fortran_COMPILER)
            unset(CMAKE_Fortran_COMPILER CACHE)
        endif()
    endif()
'@
if (-not $content.Contains($replacement)) {
    if (-not $content.Contains($needle)) {
        throw "Could not find the expected CMakeDetermineFortranCompiler hook in $finder"
    }
    Set-Content -LiteralPath $finder -Value $content.Replace($needle, $replacement) -NoNewline
}

$keptPath = [System.Collections.Generic.List[string]]::new()
foreach ($entry in ($env:Path -split ';')) {
    if ([string]::IsNullOrWhiteSpace($entry)) { continue }
    $expanded = [Environment]::ExpandEnvironmentVariables($entry.Trim('"'))
    $hasFortran = @('flang.exe', 'flang-new.exe', 'gfortran.exe', 'ifort.exe', 'ifx.exe') |
        Where-Object { Test-Path -LiteralPath (Join-Path $expanded $_) } |
        Select-Object -First 1
    if (-not $hasFortran) { $keptPath.Add($entry) }
}
Remove-Item Env:FC -ErrorAction SilentlyContinue
Remove-Item Env:F77 -ErrorAction SilentlyContinue
Remove-Item Env:F90 -ErrorAction SilentlyContinue
$env:Path = [string]::Join(';', $keptPath)

$env:VCPKG_ROOT = $VcpkgRoot
$env:VCPKG_DEFAULT_TRIPLET = $triplet
& (Join-Path $PSScriptRoot 'ensure-windows-vcpkg-dependencies.ps1') `
    -VcpkgRoot $VcpkgRoot `
    -Triplet $triplet `
    -OverlayPorts $overlayPorts

$lapackLibraries = if (Test-Path -LiteralPath (Join-Path $prefix 'lib\libf2c.lib')) {
    'lapack;libf2c;openblas'
} else {
    'lapack;openblas'
}

$environment = [ordered]@{
    VCPKG_ROOT = $VcpkgRoot
    VCPKG_DEFAULT_TRIPLET = $triplet
    VCPKGRS_TRIPLET = $triplet
    VCPKGRS_DYNAMIC = '1'
    INCLUDE = "$prefix\include;$env:INCLUDE"
    LIB = "$prefix\lib;$env:LIB"
    ZMQ_PATH = $prefix
    ZMQ_INCLUDE_DIR = "$prefix\include"
    ZMQ_LIB_DIR = "$prefix\lib"
    PKG_CONFIG_PATH = "$prefix\lib\pkgconfig"
    BLAS_LIB_DIR = "$prefix\lib"
    BLAS_LIBS = 'openblas'
    LAPACK_LIB_DIR = "$prefix\lib"
    LAPACK_LIBS = $lapackLibraries
    OPENBLAS_DIR = $prefix
    HDF5_DIR = $prefix
    RUNMAT_OCCT_ROOT = $prefix
    RUNMAT_OCCT_INCLUDE_DIR = "$prefix\include\opencascade"
    RUNMAT_OCCT_LIB_DIR = "$prefix\lib"
    RUNMAT_OCCT_BIN_DIR = "$prefix\bin"
    RUNMAT_OCCT_LINK_MODE = 'dylib'
}

if (-not $env:GITHUB_ENV -or -not $env:GITHUB_PATH) {
    throw 'GITHUB_ENV and GITHUB_PATH must be available when preparing cross-build dependencies.'
}
foreach ($entry in $environment.GetEnumerator()) {
    Add-Content -LiteralPath $env:GITHUB_ENV -Value "$($entry.Key)=$($entry.Value)"
}
Add-Content -LiteralPath $env:GITHUB_PATH -Value $VcpkgRoot
Add-Content -LiteralPath $env:GITHUB_PATH -Value (Join-Path $prefix 'bin')

Write-Host "Windows cross-build dependencies are ready under $prefix"
