param(
    [Parameter(Mandatory = $true)]
    [string]$RunMatBinary,
    [string]$RuntimePack
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $RunMatBinary -PathType Leaf)) {
    throw "RunMat binary does not exist: $RunMatBinary"
}

$smokeDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("runmat-aot-smoke-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $smokeDirectory | Out-Null

try {
    $source = Join-Path $smokeDirectory 'smoke.m'
    $program = Join-Path $smokeDirectory 'smoke.exe'
    Set-Content -LiteralPath $source -Value "x = 2 + 3;`ndisp(x);"

    if ($RuntimePack) {
        & $RunMatBinary --offline aot install --from $RuntimePack
        if ($LASTEXITCODE -ne 0) { throw "compile runtime installation failed with exit code $LASTEXITCODE" }
    }
    & $RunMatBinary --offline compile $source -o $program
    if ($LASTEXITCODE -ne 0) {
        throw "compile runtime verification failed with exit code $LASTEXITCODE"
    }

    $output = & $program
    if ($LASTEXITCODE -ne 0) {
        throw "compiled smoke program failed with exit code $LASTEXITCODE"
    }
    if (($output | Out-String).Trim() -ne '5') {
        throw "compiled smoke program returned unexpected output: $output"
    }
} finally {
    Remove-Item -LiteralPath $smokeDirectory -Recurse -Force -ErrorAction SilentlyContinue
}
