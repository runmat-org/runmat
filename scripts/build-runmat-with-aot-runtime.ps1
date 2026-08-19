$ErrorActionPreference = "Stop"

$profile = if ($env:RUNMAT_BUILD_PROFILE) { $env:RUNMAT_BUILD_PROFILE } else { "release" }
$profileDirectory = if ($profile -eq "dev") { "debug" } else { $profile }
$buildRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("runmat-aot-build-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $buildRoot | Out-Null
$target = $env:RUNMAT_BUILD_TARGET
$lockedArgs = @()
for ($index = 0; $index -lt $args.Count; $index++) {
  $argument = [string]$args[$index]
  if ($argument -eq '--target' -and ($index + 1) -lt $args.Count) {
    $target = [string]$args[$index + 1]
  } elseif ($argument.StartsWith('--target=')) {
    $target = $argument.Substring('--target='.Length)
  } elseif ($argument -in @('--locked', '--offline', '--frozen')) {
    $lockedArgs += $argument
  }
}
$targetArgs = @()
$targetDirectory = Join-Path 'target' $profileDirectory
if ($target) {
  $targetArgs = @('--target', $target)
  $targetDirectory = Join-Path 'target' (Join-Path $target $profileDirectory)
}

try {
  $rustcOutput = & cargo rustc @lockedArgs -p runmat-aot-runtime --profile $profile @targetArgs --lib --crate-type staticlib -- --print native-static-libs 2>&1
  $rustcOutput | ForEach-Object { Write-Host $_ }
  if ($LASTEXITCODE -ne 0) {
    throw "failed to build the RunMat AOT runtime archive"
  }

  $archive = Join-Path $targetDirectory "runmat_aot_runtime.lib"
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
  $packArgs = @(
    "run"
  ) + $lockedArgs + @(
    "-p", "runmat-aot", "--bin", "runmat-aot-pack", "--",
    "--archive", $archive,
    "--payload-out", $payload,
    "--manifest-out", $manifest
  )
  foreach ($token in $nativeTokens) {
    $packArgs += @("--native-link-token", $token)
  }
  & cargo @packArgs
  if ($LASTEXITCODE -ne 0) {
    throw "failed to package the RunMat AOT runtime archive"
  }

  $previousPayload = $env:RUNMAT_AOT_RUNTIME_ARCHIVE
  $previousManifest = $env:RUNMAT_AOT_RUNTIME_MANIFEST
  try {
    $env:RUNMAT_AOT_RUNTIME_ARCHIVE = $payload
    $env:RUNMAT_AOT_RUNTIME_MANIFEST = $manifest
    & cargo build -p runmat --profile $profile @args
    if ($LASTEXITCODE -ne 0) {
      throw "failed to build RunMat with the embedded AOT runtime"
    }
  } finally {
    $env:RUNMAT_AOT_RUNTIME_ARCHIVE = $previousPayload
    $env:RUNMAT_AOT_RUNTIME_MANIFEST = $previousManifest
  }
} finally {
  if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
  }
}
