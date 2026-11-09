$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'resolve-cargo-native-import-libraries.ps1')
$root = Join-Path ([IO.Path]::GetTempPath()) ('runmat-import-resolution-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
try {
    $packages = @('runtime', 'imports', 'unrelated') | ForEach-Object {
        $directory = Join-Path $root $_
        New-Item -ItemType Directory -Path (Join-Path $directory 'lib') | Out-Null
        [pscustomobject]@{id=$_;manifest_path=(Join-Path $directory 'Cargo.toml')}
    }
    $manifest = $packages[0].manifest_path
    $import = Join-Path $root 'imports/lib/windows.lib'
    [IO.File]::WriteAllText($import, 'required import members')
    [IO.File]::WriteAllText((Join-Path $root 'unrelated/lib/windows.lib'), 'unrelated target')
    $metadata = [pscustomobject]@{packages=$packages;resolve=[pscustomobject]@{nodes=@(
        [pscustomobject]@{id='runtime';deps=@([pscustomobject]@{pkg='imports'})},
        [pscustomobject]@{id='imports';deps=@()},
        [pscustomobject]@{id='unrelated';deps=@()}
    )}}
    $found = @(Resolve-CargoNativeImportLibraries $metadata $manifest @('windows.lib','kernel32.lib','windows.lib'))
    if ($found.Count -ne 1 -or $found[0] -ne [IO.Path]::GetFullPath($import)) { throw 'Runtime dependency closure or token conservation failed' }
    $metadata.resolve.nodes[0].deps += [pscustomobject]@{pkg='unrelated'}
    $rejected = $false
    try { Resolve-CargoNativeImportLibraries $metadata $manifest @('windows.lib') | Out-Null }
    catch { if ($_.Exception.Message -notlike '*conflicting import libraries*') { throw }; $rejected=$true }
    if (-not $rejected) { throw 'Conflicting import-library identities were accepted' }
    [IO.File]::WriteAllText((Join-Path $root 'unrelated/lib/windows.lib'), 'required import members')
    $found = @(Resolve-CargoNativeImportLibraries $metadata $manifest @('windows.lib','kernel32.lib'))
    if ($found.Count -ne 1) { throw 'Byte-identical libraries were not deduplicated' }
    Write-Host 'Runtime import-library closure, identity conflicts and system-token conservation passed.'
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force
}
