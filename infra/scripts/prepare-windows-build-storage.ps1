#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$TargetDirectory,
    [Parameter(Mandatory = $true)][string]$ScratchDirectory
)
$ErrorActionPreference = 'Stop'

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
[pscustomobject]@{TargetDirectory=$target; ScratchDirectory=$scratch}
