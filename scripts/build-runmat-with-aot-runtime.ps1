$ErrorActionPreference = "Stop"

$profile = if ($env:RUNMAT_BUILD_PROFILE) { $env:RUNMAT_BUILD_PROFILE } else { "release" }
$buildRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("runmat-aot-build-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$target = if ($env:RUNMAT_BUILD_TARGET) { $env:RUNMAT_BUILD_TARGET } else { $env:CARGO_BUILD_TARGET }
$lockedArgs = @()
$buildArgs = @()
$runtimeNoDefault = $false
$runtimeWgpu = $false
$runtimeCuda = $false
for ($index = 0; $index -lt $args.Count; $index++) {
  $argument = [string]$args[$index]
  if ($argument -eq '--release') {
    $profile = 'release'
  } elseif ($argument -eq '--profile') {
    if (($index + 1) -ge $args.Count) { throw '--profile requires a value' }
    $index++
    $profile = [string]$args[$index]
  } elseif ($argument.StartsWith('--profile=')) {
    $profile = $argument.Substring('--profile='.Length)
  } elseif ($argument -eq '--target') {
    if (($index + 1) -ge $args.Count) { throw '--target requires a value' }
    $index++
    $target = [string]$args[$index]
    if (-not $target) { throw '--target requires a value' }
  } elseif ($argument.StartsWith('--target=')) {
    $target = $argument.Substring('--target='.Length)
    if (-not $target) { throw '--target requires a value' }
  } elseif ($argument -in @('--locked', '--offline', '--frozen')) {
    $lockedArgs += $argument
    $buildArgs += $argument
  } elseif ($argument -eq '--no-default-features') {
    $runtimeNoDefault = $true
    $buildArgs += $argument
  } elseif ($argument -eq '--features' -or $argument.StartsWith('--features=')) {
    if ($argument -eq '--features') {
      if (($index + 1) -ge $args.Count) { throw '--features requires a value' }
      $index++
      $featureText = [string]$args[$index]
      $buildArgs += @($argument, $featureText)
    } else {
      $featureText = $argument.Substring('--features='.Length)
      $buildArgs += $argument
    }
    foreach ($feature in ($featureText -split '[,\s]+')) {
      if ($feature -in @('wgpu', 'runmat/wgpu')) { $runtimeWgpu = $true }
      if ($feature -in @('cuda', 'runmat/cuda')) { $runtimeCuda = $true }
    }
  } elseif ($argument -eq '--all-features') {
    $runtimeWgpu = $true
    $runtimeCuda = $true
    $buildArgs += $argument
  } else {
    $buildArgs += $argument
  }
}
$profileDirectory = if ($profile -eq "dev") { "debug" } else { $profile }
$targetArgs = @()
$targetRoot = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target' }
$targetDirectory = Join-Path $targetRoot $profileDirectory
if ($target) {
  $targetArgs = @('--target', $target)
  $targetDirectory = Join-Path $targetRoot (Join-Path $target $profileDirectory)
}

$previousMacosMinimum = [System.Environment]::GetEnvironmentVariable('MACOSX_DEPLOYMENT_TARGET', 'Process')
$previousCmakeMinimum = [System.Environment]::GetEnvironmentVariable('CMAKE_OSX_DEPLOYMENT_TARGET', 'Process')
$darwinPolicyNeeded = $false
try {
  # This tool reads the deployment archive but must execute on the build host,
  # even when Cargo's environment selects a different deployment target.
  $rustcCommand = if ($env:RUSTC) { $env:RUSTC } else { 'rustc' }
  $compilerMetadata = & $rustcCommand -vV
  if ($LASTEXITCODE -ne 0) { throw 'failed to inspect the build-host Rust compiler' }
  $hostLine = $compilerMetadata | Where-Object { $_ -match '^host: (\S+)$' } | Select-Object -Last 1
  if (-not $hostLine) { throw 'Rust compiler did not report its build-host target' }
  $hostTarget = ([regex]::Match($hostLine, '^host: (\S+)$')).Groups[1].Value
  $effectiveTarget = if ($target) { $target } else { $hostTarget }
  $targetIsDarwin = $effectiveTarget.EndsWith('-apple-darwin')
  $hostIsDarwin = $hostTarget.EndsWith('-apple-darwin')
  $darwinPolicyNeeded = $targetIsDarwin -or $hostIsDarwin
  if ($darwinPolicyNeeded) {
    $macosMinimum = $previousMacosMinimum
    $cmakeMinimum = $previousCmakeMinimum
    if ($macosMinimum -and $cmakeMinimum -and
        (($macosMinimum -replace '(\.0)+$', '') -ne ($cmakeMinimum -replace '(\.0)+$', ''))) {
      throw 'conflicting MACOSX_DEPLOYMENT_TARGET and CMAKE_OSX_DEPLOYMENT_TARGET'
    }
    $deploymentMinimum = if ($macosMinimum) { $macosMinimum } elseif ($cmakeMinimum) { $cmakeMinimum } else { '11.0' }
    if ($targetIsDarwin) {
      [System.Environment]::SetEnvironmentVariable('MACOSX_DEPLOYMENT_TARGET', $deploymentMinimum, 'Process')
      [System.Environment]::SetEnvironmentVariable('CMAKE_OSX_DEPLOYMENT_TARGET',
        $(if ($cmakeMinimum) { $cmakeMinimum } else { $deploymentMinimum }), 'Process')
    }
  }
  $runtimeFeatureArgs = @()
  if ($runtimeNoDefault) { $runtimeFeatureArgs += '--no-default-features' }
  # The stock CLI keeps console/callable factories in CPU-only builds too.
  # Custom embedding archives select their own factory symbols directly.
  $runtimeFeatures = @('stock-hosted-factories')
  if ($runtimeWgpu) { $runtimeFeatures += 'wgpu' }
  if ($runtimeCuda) { $runtimeFeatures += 'cuda' }
  $runtimeFeatureArgs += @('--features', ($runtimeFeatures -join ','))
  # This output becomes machine-readable input to the packer. Prevent an
  # ambient CARGO_TERM_COLOR setting from adding ANSI sequences to the
  # native-static-libs line, then restore the caller's preference.
  $previousCargoTermColor = $env:CARGO_TERM_COLOR
  # This phase produces relocatable archive objects, not the finished CLI.
  # Preserve their symbols throughout the dependency closure and restore the
  # caller's stripping policy before packaging and the final executable build.
  $runtimeStripVariable = "CARGO_PROFILE_$($profile.ToUpperInvariant().Replace('-', '_'))_STRIP"
  $previousRuntimeStrip = [System.Environment]::GetEnvironmentVariable($runtimeStripVariable, 'Process')
  try {
    $env:CARGO_TERM_COLOR = 'never'
    [System.Environment]::SetEnvironmentVariable($runtimeStripVariable, 'none', 'Process')
    $rustcOutput = & cargo rustc @lockedArgs -p runmat-aot-runtime --profile $profile @targetArgs @runtimeFeatureArgs --lib --crate-type staticlib -- --print native-static-libs 2>&1
    $rustcExitCode = $LASTEXITCODE
  } finally {
    $env:CARGO_TERM_COLOR = $previousCargoTermColor
    [System.Environment]::SetEnvironmentVariable($runtimeStripVariable, $previousRuntimeStrip, 'Process')
  }
  $rustcOutput | ForEach-Object { Write-Host $_ }
  if ($rustcExitCode -ne 0) {
    throw "failed to build the RunMat AOT runtime archive"
  }

  # Query the deployment compiler rather than assuming the build host's format.
  $archiveNames = @('' | & $rustcCommand --crate-name runmat_aot_runtime --crate-type staticlib --print file-names @targetArgs -)
  if ($LASTEXITCODE -ne 0 -or $archiveNames.Count -ne 1 -or -not $archiveNames[0]) {
    throw 'Rust compiler did not report one AOT runtime archive name'
  }
  $archive = Join-Path $targetDirectory ([string]$archiveNames[0])
  if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "AOT runtime archive was not produced at $archive"
  }

  $nativeLine = $rustcOutput |
    ForEach-Object { [string]$_ } |
    Where-Object { $_ -match '^note: native-static-libs: (.*)$' } |
    Select-Object -Last 1
  if (-not $nativeLine) {
    throw "Cargo did not report native-static-libs for the AOT runtime archive"
  }
  $nativeTokens = ([regex]::Match($nativeLine, '^note: native-static-libs: (.*)$').Groups[1].Value -split '\s+') |
    Where-Object { $_ -ne "" }

  $payload = Join-Path $buildRoot "runtime-archive.payload"
  $manifest = Join-Path $buildRoot "runtime-archive.json"
  $notices = Join-Path $buildRoot "THIRD_PARTY_NOTICES.txt"
  $packArgs = @(
    "run", "--profile", $profile
  ) + $lockedArgs + @('--target', $hostTarget) + @(
    "-p", "runmat-aot", "--bin", "runmat-aot-pack", "--",
    "--archive", $archive,
    "--payload-out", $payload,
    "--manifest-out", $manifest,
    "--notices-out", $notices
  )
  foreach ($token in $nativeTokens) {
    $packArgs += @("--native-link-token", $token)
  }
  if ($effectiveTarget.EndsWith('-windows-msvc')) {
    # Older Rust Windows bindings reference import libraries held in Cargo's
    # registry. Resolve them from this runtime dependency graph and bundle them.
    . (Join-Path $PSScriptRoot 'distribution/resolve-cargo-native-import-libraries.ps1')
    $runtimeManifest = Join-Path $PSScriptRoot '../crates/runmat-aot-runtime/Cargo.toml'
    $metadataArgs = @('metadata', '--format-version', '1', '--manifest-path', $runtimeManifest,
      '--filter-platform', $effectiveTarget) + $lockedArgs
    if ($runtimeNoDefault) { $metadataArgs += '--no-default-features' }
    $metadataFeatures = @($runtimeFeatures | ForEach-Object { "runmat-aot-runtime/$_" })
    $metadataArgs += @('--features', ($metadataFeatures -join ','))
    $metadataOutput = & cargo @metadataArgs
    if ($LASTEXITCODE -ne 0) { throw 'failed to resolve runtime import-library provenance' }
    $metadata = ($metadataOutput -join "`n") | ConvertFrom-Json -Depth 100
    $importLibraries = @(Resolve-CargoNativeImportLibraries -Metadata $metadata `
      -RuntimeManifestPath $runtimeManifest -NativeLinkTokens $nativeTokens)
    foreach ($library in $importLibraries) { $packArgs += @('--msvc-import-library', $library) }
  }
  # Select the host-tool policy independently from the deployment policy.
  if ($darwinPolicyNeeded) {
    [System.Environment]::SetEnvironmentVariable('MACOSX_DEPLOYMENT_TARGET',
      $(if ($hostIsDarwin) { $deploymentMinimum } else { $previousMacosMinimum }), 'Process')
    [System.Environment]::SetEnvironmentVariable('CMAKE_OSX_DEPLOYMENT_TARGET',
      $(if ($hostIsDarwin) {
        if ($cmakeMinimum) { $cmakeMinimum } else { $deploymentMinimum }
      } else { $previousCmakeMinimum }), 'Process')
  }
  try {
    & cargo @packArgs
    $packExitCode = $LASTEXITCODE
  } finally {
    if ($darwinPolicyNeeded) {
      [System.Environment]::SetEnvironmentVariable('MACOSX_DEPLOYMENT_TARGET',
        $(if ($targetIsDarwin) { $deploymentMinimum } else { $previousMacosMinimum }), 'Process')
      [System.Environment]::SetEnvironmentVariable('CMAKE_OSX_DEPLOYMENT_TARGET',
        $(if ($targetIsDarwin) {
          if ($cmakeMinimum) { $cmakeMinimum } else { $deploymentMinimum }
        } else { $previousCmakeMinimum }), 'Process')
    }
  }
  if ($packExitCode -ne 0) {
    throw "failed to package the RunMat AOT runtime archive"
  }

  $previousPayload = $env:RUNMAT_AOT_RUNTIME_ARCHIVE
  $previousManifest = $env:RUNMAT_AOT_RUNTIME_MANIFEST
  try {
    $env:RUNMAT_AOT_RUNTIME_ARCHIVE = $payload
    $env:RUNMAT_AOT_RUNTIME_MANIFEST = $manifest
    & cargo build -p runmat --profile $profile @targetArgs @buildArgs
    if ($LASTEXITCODE -ne 0) {
      throw "failed to build RunMat with the embedded AOT runtime"
    }
    $noticeDestination = Join-Path $targetDirectory "THIRD_PARTY_NOTICES.txt"
    $existing = Get-Item -LiteralPath $noticeDestination -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
      if (-not ($existing -is [System.IO.FileInfo]) -or ($existing.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -or
          (Get-FileHash -LiteralPath $notices -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $noticeDestination -Algorithm SHA256).Hash) {
        throw "existing AOT notice differs from canonical Runtime bytes: $noticeDestination"
      }
    } else {
      [System.IO.File]::Copy($notices, $noticeDestination, $false)
    }
  } finally {
    $env:RUNMAT_AOT_RUNTIME_ARCHIVE = $previousPayload
    $env:RUNMAT_AOT_RUNTIME_MANIFEST = $previousManifest
  }
} finally {
  if ($darwinPolicyNeeded) {
    [System.Environment]::SetEnvironmentVariable('MACOSX_DEPLOYMENT_TARGET', $previousMacosMinimum, 'Process')
    [System.Environment]::SetEnvironmentVariable('CMAKE_OSX_DEPLOYMENT_TARGET', $previousCmakeMinimum, 'Process')
  }
  if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
  }
}
