#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$TargetDirectory,
    [Parameter(Mandatory = $true)][string]$ScratchDirectory,
    [switch]$ReclaimUnusedRunnerTools
)
$ErrorActionPreference = 'Stop'

if ($ReclaimUnusedRunnerTools) {
    if ($env:GITHUB_ACTIONS -ne 'true') {
        throw 'Runner image cleanup is restricted to the isolated GitHub Actions job'
    }
    # These hosted-image tools are unused by RunMat's Rust/native jobs. Preserve
    # Visual Studio, Windows SDKs, LLVM, Rust, Python, Java, Node and CMake.
    $unused = @(
        'C:\Android\android-sdk',
        'C:\ghcup',
        'C:\Miniconda',
        'C:\hostedtoolcache\windows\CodeQL',
        'C:\hostedtoolcache\windows\go',
        'C:\hostedtoolcache\windows\PyPy',
        'C:\hostedtoolcache\windows\Ruby'
    )
    foreach ($path in $unused) {
        if (!(Test-Path -LiteralPath $path)) { continue }
        $item = Get-Item -LiteralPath $path -Force
        if (!$item.PSIsContainer -or $item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Runner tool cleanup refuses a redirected/non-directory target: $path"
        }
        for ($parent = $item.Parent; $null -ne $parent; $parent = $parent.Parent) {
            if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Runner tool cleanup refuses a redirected parent: $path"
            }
        }
        Write-Host "Removing unused ephemeral runner tool: $path"
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    # The job never uses containers. Docker owns removal of its unused image
    # layers; do not traverse or remove Docker's internal storage ourselves.
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        $containers = & docker ps --all --quiet 2>&1
        if ($LASTEXITCODE -eq 0 -and !$containers) {
            & docker image prune --all --force
            if ($LASTEXITCODE -ne 0) { throw 'Unused runner Docker image cleanup failed' }
        } else {
            Write-Host 'Docker image cleanup skipped: containers exist or daemon is unavailable'
        }
    }
}

function Enable-BuildCompression([string]$Path) {
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (![IO.Path]::GetDirectoryName($fullPath)) {
        throw 'Build compression requires a named directory, not a volume root'
    }
    $directory = New-Item -ItemType Directory -Force -Path $fullPath
    if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Build compression refuses reparse points: $fullPath"
    }
    $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($fullPath))
    if ($drive.DriveFormat -ne 'NTFS') {
        throw "Build storage requires NTFS compression support: $fullPath"
    }
    # Set the inherited storage policy before cache extraction or compilation.
    # Logical file bytes and compiler/linker inputs stay intact.
    & "$env:SystemRoot/System32/compact.exe" /C /I /Q $fullPath | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "NTFS compression failed: $fullPath" }
    $directory = Get-Item -LiteralPath $fullPath
    if (!($directory.Attributes -band [IO.FileAttributes]::Compressed)) {
        throw "Build storage was not marked compressed: $fullPath"
    }
    Write-Host ('Build volume {0}: {1:N1} GiB free of {2:N1} GiB; compressed directory {3}' -f
        $drive.Name, ($drive.AvailableFreeSpace / 1GB), ($drive.TotalSize / 1GB), $fullPath)
    return $fullPath
}

$target = Enable-BuildCompression $TargetDirectory
$scratch = Enable-BuildCompression $ScratchDirectory
if ($ReclaimUnusedRunnerTools) {
    $drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($target))
    # Reserve twice the largest observed 48-GiB batch before restoring caches.
    if ($drive.AvailableFreeSpace -lt 96GB) {
        throw ('Isolated Windows build needs 96 GiB free before cache restore; {0:N1} GiB available' -f
            ($drive.AvailableFreeSpace / 1GB))
    }
}
[pscustomobject]@{TargetDirectory=$target; ScratchDirectory=$scratch}
