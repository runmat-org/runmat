# GCE startup script (windows-startup-script-ps1): unpack the setup bundle from
# metadata and run setup.ps1. Output goes to the serial console and C:\ci\setup.log.
$ErrorActionPreference = 'Stop'
$bundle = Invoke-RestMethod -UseBasicParsing -Headers @{ 'Metadata-Flavor' = 'Google' } `
    -Uri 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/ci-bundle'
$archive = "$env:TEMP\ci-bundle.tgz"
[IO.File]::WriteAllBytes($archive, [Convert]::FromBase64String(($bundle -replace '\s', '')))
if (Test-Path C:\ci.new) { Remove-Item -Recurse -Force C:\ci.new }
New-Item -ItemType Directory -Path C:\ci.new | Out-Null
& tar.exe -xzf $archive -C C:\ci.new
New-Item -ItemType Directory -Force -Path C:\ci | Out-Null
foreach ($name in 'common', 'windows') {
    if (Test-Path "C:\ci\$name") { Remove-Item -Recurse -Force "C:\ci\$name" }
    Move-Item "C:\ci.new\$name" "C:\ci\$name"
}
Remove-Item -Recurse -Force C:\ci.new
$ErrorActionPreference = 'Continue'
& C:\ci\windows\setup.ps1 *>&1 | Tee-Object -FilePath C:\ci\setup.log -Append
