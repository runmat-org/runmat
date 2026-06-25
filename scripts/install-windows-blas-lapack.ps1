param(
  [Parameter(Mandatory = $true)]
  [string]$VcpkgRoot,

  [string]$VcpkgRef = "a7eda31dc16994fcaa8587982eb833a8695f1b6f",

  [string]$Triplet = "x64-windows",

  [switch]$AllowUnsupportedLapack,

  [switch]$ExportGitHubEnvironment
)

$ErrorActionPreference = "Stop"
$env:VCPKG_ROOT = $VcpkgRoot

function Invoke-CheckedNative {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [scriptblock]$Command
  )

  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
}

function Ensure-VcpkgCheckout {
  if (-not (Test-Path -LiteralPath $VcpkgRoot)) {
    Invoke-CheckedNative "git clone vcpkg" {
      git clone --depth 1 https://github.com/microsoft/vcpkg.git $VcpkgRoot
    }
  }

  Push-Location $VcpkgRoot
  try {
    Invoke-CheckedNative "git fetch vcpkg ref" {
      git fetch --depth 1 origin $VcpkgRef
    }
    Invoke-CheckedNative "git checkout vcpkg ref" {
      git checkout --force FETCH_HEAD
    }
  } finally {
    Pop-Location
  }

  Invoke-CheckedNative "bootstrap vcpkg" {
    & "$VcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
  }
}

function Force-VcpkgBundledGfortran {
  $finder = Join-Path $VcpkgRoot "scripts\cmake\vcpkg_find_fortran.cmake"
  if (-not (Test-Path -LiteralPath $finder)) {
    throw "Missing vcpkg Fortran finder helper at $finder"
  }

  $marker = "RUNMAT: force vcpkg internal MinGW gfortran on Windows"
  $content = Get-Content -LiteralPath $finder -Raw
  if ($content.Contains($marker)) {
    return
  }

  $needle = "    include(CMakeDetermineFortranCompiler)"
  if (-not $content.Contains($needle)) {
    throw "Could not find expected CMakeDetermineFortranCompiler hook in $finder"
  }

  $replacement = @'
    include(CMakeDetermineFortranCompiler)

    # RUNMAT: force vcpkg internal MinGW gfortran on Windows.
    # windows-latest can expose Visual Studio LLVM Flang; vcpkg's fallback
    # only uses bundled MinGW gfortran when no Fortran compiler is detected.
    if(CMAKE_HOST_WIN32 AND "${VCPKG_CHAINLOAD_TOOLCHAIN_FILE}" STREQUAL "" AND
       ("${VCPKG_TARGET_ARCHITECTURE}" STREQUAL "x86" OR "${VCPKG_TARGET_ARCHITECTURE}" STREQUAL "x64"))
        if(CMAKE_Fortran_COMPILER)
            message(STATUS "Ignoring ambient Fortran compiler '${CMAKE_Fortran_COMPILER}' so vcpkg uses internal MinGW gfortran")
            unset(CMAKE_Fortran_COMPILER)
            unset(CMAKE_Fortran_COMPILER CACHE)
        endif()
    endif()
'@

  Set-Content -LiteralPath $finder -Value $content.Replace($needle, $replacement) -NoNewline
}

function Remove-AmbientFortranCompilersFromPath {
  $compilerNames = @(
    "flang.exe",
    "flang-new.exe",
    "gfortran.exe",
    "ifort.exe",
    "ifx.exe"
  )

  $kept = [System.Collections.Generic.List[string]]::new()

  foreach ($entry in ($env:Path -split ";")) {
    if ([string]::IsNullOrWhiteSpace($entry)) {
      continue
    }

    $trimmed = $entry.Trim()
    $expanded = [Environment]::ExpandEnvironmentVariables($trimmed.Trim('"'))
    $containsFortranCompiler = $false

    foreach ($compilerName in $compilerNames) {
      if (Test-Path -LiteralPath (Join-Path $expanded $compilerName)) {
        $containsFortranCompiler = $true
        break
      }
    }

    if ($containsFortranCompiler) {
      Write-Host "Temporarily removing ambient Fortran compiler PATH entry: $trimmed"
    } else {
      $kept.Add($trimmed)
    }
  }

  Remove-Item Env:FC -ErrorAction SilentlyContinue
  Remove-Item Env:F77 -ErrorAction SilentlyContinue
  Remove-Item Env:F90 -ErrorAction SilentlyContinue
  $env:Path = [string]::Join(";", $kept)
}

function Assert-InstalledArtifacts {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Prefix
  )

  $requiredFiles = @(
    (Join-Path $Prefix "lib\openblas.lib"),
    (Join-Path $Prefix "lib\lapack.lib"),
    (Join-Path $Prefix "bin\libgfortran-5.dll"),
    (Join-Path $Prefix "bin\libquadmath-0.dll"),
    (Join-Path $Prefix "bin\libwinpthread-1.dll"),
    (Join-Path $Prefix "bin\libgcc_s_seh-1.dll")
  )

  foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile)) {
      throw "Expected Windows BLAS/LAPACK artifact not found: $requiredFile"
    }
  }
}

function Export-GitHubEnvironment {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Prefix
  )

  if (-not $env:GITHUB_ENV) {
    throw "GITHUB_ENV is not set; cannot export vcpkg environment."
  }
  if (-not $env:GITHUB_PATH) {
    throw "GITHUB_PATH is not set; cannot export vcpkg PATH."
  }

  Add-Content -Path $env:GITHUB_ENV -Value "VCPKG_ROOT=$VcpkgRoot"
  Add-Content -Path $env:GITHUB_ENV -Value "VCPKG_DEFAULT_TRIPLET=$Triplet"
  Add-Content -Path $env:GITHUB_ENV -Value "VCPKGRS_TRIPLET=$Triplet"
  Add-Content -Path $env:GITHUB_ENV -Value "VCPKGRS_DYNAMIC=1"
  Add-Content -Path $env:GITHUB_ENV -Value "OPENBLAS_DIR=$Prefix"
  Add-Content -Path $env:GITHUB_ENV -Value "BLAS_LIB_DIR=$Prefix\lib"
  Add-Content -Path $env:GITHUB_ENV -Value "BLAS_LIBS=openblas"
  Add-Content -Path $env:GITHUB_ENV -Value "LAPACK_LIB_DIR=$Prefix\lib"
  Add-Content -Path $env:GITHUB_ENV -Value "LAPACK_LIBS=lapack;openblas"
  Add-Content -Path $env:GITHUB_PATH -Value "$Prefix\bin"
}

Ensure-VcpkgCheckout
Force-VcpkgBundledGfortran
Remove-AmbientFortranCompilersFromPath

$vcpkgExe = Join-Path $VcpkgRoot "vcpkg.exe"
$prefix = Join-Path $VcpkgRoot "installed\$Triplet"
$lapackArgs = @("install", "openblas:$Triplet", "lapack-reference:$Triplet")
if ($AllowUnsupportedLapack) {
  $lapackArgs += "--allow-unsupported"
}

Invoke-CheckedNative "vcpkg install Windows BLAS/LAPACK dependencies" {
  & $vcpkgExe @lapackArgs
}

Assert-InstalledArtifacts -Prefix $prefix

if ($ExportGitHubEnvironment) {
  Export-GitHubEnvironment -Prefix $prefix
}

Write-Host "Windows BLAS/LAPACK dependencies are installed with vcpkg bundled MinGW gfortran runtime."
