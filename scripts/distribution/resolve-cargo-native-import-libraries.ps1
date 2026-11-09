function Resolve-CargoNativeImportLibraries {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]$Metadata,
        [Parameter(Mandatory=$true)][string]$RuntimeManifestPath,
        [Parameter(Mandatory=$true)][string[]]$NativeLinkTokens
    )
    $packages = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($package in $Metadata.packages) {
        if (-not $package.id -or -not $package.manifest_path -or -not $packages.TryAdd($package.id, $package)) {
            throw 'Cargo metadata contains an invalid or duplicate package'
        }
    }
    $nodes = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($node in $Metadata.resolve.nodes) {
        if (-not $node.id -or -not $nodes.TryAdd($node.id, $node)) {
            throw 'Cargo metadata contains an invalid or duplicate resolution node'
        }
    }
    $manifest = [IO.Path]::GetFullPath($RuntimeManifestPath)
    $roots = @($packages.Values | Where-Object { [IO.Path]::GetFullPath($_.manifest_path) -eq $manifest })
    if ($roots.Count -ne 1) { throw 'Cargo metadata must identify one runtime archive package' }
    $pending = [Collections.Generic.Queue[string]]::new()
    $pending.Enqueue($roots[0].id)
    $reachable = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    while ($pending.Count) {
        $id = $pending.Dequeue()
        if (-not $reachable.Add($id)) { continue }
        if (-not $nodes.ContainsKey($id) -or -not $packages.ContainsKey($id)) {
            throw "Cargo metadata lacks the runtime dependency $id"
        }
        foreach ($dependency in $nodes[$id].deps) { $pending.Enqueue($dependency.pkg) }
    }
    foreach ($token in @($NativeLinkTokens | Sort-Object -Unique)) {
        if ([IO.Path]::GetFileName($token) -ne $token -or [IO.Path]::GetExtension($token) -ne '.lib') { continue }
        $candidates = @(
            foreach ($id in $reachable) {
                $candidate = Join-Path (Split-Path $packages[$id].manifest_path -Parent) (Join-Path 'lib' $token)
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { [IO.Path]::GetFullPath($candidate) }
            }
        ) | Sort-Object -Unique
        $candidates = @($candidates)
        if (-not $candidates.Count) { continue } # SDK/system libraries retain their link tokens.
        $hashes = @($candidates | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash } | Sort-Object -Unique)
        if ($hashes.Count -ne 1) { throw "Runtime dependencies provide conflicting import libraries for $token" }
        $candidates[0]
    }
}
