#Requires -Version 7.0
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc')]
    [string]$Target,
    [Parameter(Mandatory = $true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$descriptor = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'windows-dxc-runtime.json') -Raw | ConvertFrom-Json
if ($descriptor.schema_version -ne 1) { throw 'Unsupported DXC descriptor schema' }
$selection = $descriptor.targets.$Target
$destinationPath = [IO.Path]::GetFullPath($Destination)
$parent = [IO.Path]::GetDirectoryName($destinationPath)
if (!$parent -or ![IO.Path]::GetFileName($destinationPath)) { throw 'DXC destination must be a named artifact directory' }
$files = @(
    foreach ($dll in $selection.dlls.PSObject.Properties) {
        [pscustomobject]@{Path="bin/$($dll.Name)"; ArchivePath="bin/$($selection.architecture)/$($dll.Name)"; Sha256=$dll.Value}
    }
    foreach ($notice in $descriptor.notices.PSObject.Properties) {
        [pscustomobject]@{Path="notices/directx-dxc-$($notice.Name)"; ArchivePath=$notice.Name; Sha256=$notice.Value}
    }
)

function Assert-DxcArtifact([string]$Root) {
    $directory = Get-Item -LiteralPath $Root -Force
    if (!$directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'DXC artifact must be a real directory' }
    $actual = @(Get-ChildItem -LiteralPath $Root -Recurse -Force)
    if (@($actual | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count) { throw 'DXC artifact contains a reparse point' }
    if (@($actual | Where-Object {!$_.PSIsContainer}).Count -ne $files.Count + 1) { throw 'DXC artifact file inventory differs from pinned inputs' }
    foreach ($file in $files) {
        $path = Join-Path $Root $file.Path
        if (!(Test-Path -LiteralPath $path -PathType Leaf) -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Sha256) { throw "DXC pinned-file mismatch: $($file.Path)" }
    }
    $receipt = Get-Content -LiteralPath (Join-Path $Root 'artifact.json') -Raw | ConvertFrom-Json
    if ($receipt.schema_version -ne 1 -or $receipt.target -ne $Target -or $receipt.release -ne $descriptor.release -or $receipt.archive_sha512 -ne $descriptor.archive_sha512) { throw 'DXC artifact provenance differs from pinned descriptor' }
    if (@($receipt.files).Count -ne $files.Count) { throw 'DXC receipt file inventory differs from pinned descriptor' }
    for ($index = 0; $index -lt $files.Count; $index++) {
        $actualFile = $receipt.files[$index]
        $expectedFile = $files[$index]
        if ($actualFile.Path -cne $expectedFile.Path -or $actualFile.ArchivePath -cne $expectedFile.ArchivePath -or $actualFile.Sha256 -cne $expectedFile.Sha256) { throw 'DXC receipt file identity differs from pinned descriptor' }
    }
}

if (!(Test-Path -LiteralPath $destinationPath)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    # Only this newly created, uniquely named scratch directory is cleaned.
    $scratch = Join-Path $parent ('.runmat-dxc-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    try {
        $archive = Join-Path $scratch 'publisher.zip'
        try {
            Invoke-WebRequest -Uri $descriptor.url -OutFile $archive
        } catch {
            $node = Get-Command node -ErrorAction SilentlyContinue
            $gh = Get-Command gh -ErrorAction SilentlyContinue
            if (!$node -or !$gh) { throw }
            # Retain the pinned archive identity when the release download route
            # is unavailable. The shared provider verifies SHA512 before staging.
            Write-Host 'DXC release download failed; trying the GitHub asset API'
            Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
            $provider = Join-Path $PSScriptRoot '../../infra/scripts/github-release-asset-cache.mjs'
            & $node.Source $provider $descriptor.url $descriptor.archive_sha512 $archive $gh.Source
            if ($LASTEXITCODE -ne 0) { throw "DXC asset API download failed with exit code $LASTEXITCODE" }
        }
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA512).Hash.ToLowerInvariant() -ne $descriptor.archive_sha512) { throw 'DXC publisher archive checksum mismatch' }
        $unpacked = Join-Path $scratch 'unpacked'
        Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
        $payload = Join-Path $scratch 'payload'
        New-Item -ItemType Directory -Path (Join-Path $payload 'bin'),(Join-Path $payload 'notices') | Out-Null
        foreach ($file in $files) {
            Copy-Item -LiteralPath (Join-Path $unpacked $file.ArchivePath) -Destination (Join-Path $payload $file.Path)
        }
        [ordered]@{schema_version=1; target=$Target; release=$descriptor.release; archive_sha512=$descriptor.archive_sha512; files=$files} |
            ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath (Join-Path $payload 'artifact.json') -Encoding utf8NoBOM -NoNewline
        Assert-DxcArtifact $payload
        [IO.Directory]::Move($payload, $destinationPath)
    } finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
}
# Reuse is verified against source-controlled publisher pins, not hashes supplied
# by a mutable cache manifest. Unexpected or damaged cache content is refused.
Assert-DxcArtifact $destinationPath
[pscustomobject]@{Root=$destinationPath; BinDir=(Join-Path $destinationPath 'bin'); NoticeDir=(Join-Path $destinationPath 'notices'); Release=$descriptor.release; Target=$Target}
