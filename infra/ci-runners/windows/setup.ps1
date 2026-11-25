#Requires -Version 5.1
# Provision or refresh a Windows runner VM. startup.ps1 runs this as SYSTEM at
# every boot; every step is idempotent.
# Native tools (choco, rustup, config.cmd) write progress to stderr, which
# Windows PowerShell turns into errors under 'Stop'. Cmdlets that must succeed
# pass -ErrorAction Stop; native calls check $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$CI = 'C:\ci'

function Get-Md([string]$Key) {
    try {
        # Invoke-WebRequest keeps the value as text; Invoke-RestMethod would parse JSON values.
        $content = (Invoke-WebRequest -Headers @{ 'Metadata-Flavor' = 'Google' } -UseBasicParsing `
            -Uri "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$Key").Content
        if ($content -is [byte[]]) { $content = [Text.Encoding]::UTF8.GetString($content) }
        return [string]$content
    } catch { return '' }
}
$slots = (Get-Md 'ci-slots') -split '\s+' | Where-Object { $_ }
# Runner names use the instance name: Windows cuts the computer name to 15
# characters (ci-windows-build-x64 becomes CI-WINDOWS-BUIL).
try {
    $instance = (Invoke-WebRequest -Headers @{ 'Metadata-Flavor' = 'Google' } -UseBasicParsing `
        -Uri 'http://metadata.google.internal/computeMetadata/v1/instance/name').Content
    if ($instance -is [byte[]]) { $instance = [Text.Encoding]::UTF8.GetString($instance) }
    $instance = [string]$instance
} catch { $instance = '' }
if (-not $instance) { $instance = $env:COMPUTERNAME }
$instance = $instance.Trim().ToLower()
$label = Get-Md 'ci-label'
$runnerVersion = Get-Md 'ci-runner-version'
$idleMinutes = Get-Md 'ci-idle-minutes'
if (-not $idleMinutes) { $idleMinutes = 30 }

# ---- disks: work (D:) and cache (E:) are separate disks, so a full cache or a
# runaway job cannot fill C:. They are told apart by size. On a VM whose work and
# cache are local SSDs (ci-windows-build-x64), both come back blank after every
# stop, and this step formats them again at boot.
foreach ($spec in @(@{ Letter = 'D'; Label = 'ci-work'; MaxGB = 1000 }, @{ Letter = 'E'; Label = 'ci-cache'; MaxGB = 100000 })) {
    if (Get-Volume -DriveLetter $spec.Letter -ErrorAction SilentlyContinue) { continue }
    $existing = Get-Volume -FileSystemLabel $spec.Label -ErrorAction SilentlyContinue
    if ($existing) {
        Get-Partition -Volume $existing | Set-Partition -NewDriveLetter $spec.Letter
        continue
    }
    $raw = Get-Disk | Where-Object { $_.PartitionStyle -eq 'RAW' -and $_.Size -gt 200GB } | Sort-Object Size
    $disk = if ($spec.Label -eq 'ci-work') { $raw | Select-Object -First 1 } else { $raw | Select-Object -Last 1 }
    if (-not $disk) { throw "No raw disk for $($spec.Label)" }
    Initialize-Disk -Number $disk.Number -PartitionStyle GPT -ErrorAction Stop
    New-Partition -DiskNumber $disk.Number -UseMaximumSize -DriveLetter $spec.Letter -ErrorAction Stop |
        Format-Volume -FileSystem NTFS -NewFileSystemLabel $spec.Label -AllocationUnitSize 4096 -Confirm:$false -ErrorAction Stop | Out-Null
}

# ---- system settings and tools (first boot only)
$state = 'C:\ci\state'
New-Item -ItemType Directory -Force -Path $state | Out-Null
if (-not (Test-Path "$state\packages-v1")) {
    Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1
    # Real-time scanning of build trees costs more than the build; scan the system disk only.
    Add-MpPreference -ExclusionPath 'D:\', 'E:\', 'C:\ci-runner', 'C:\ci'
    Add-MpPreference -ExclusionProcess 'cargo.exe', 'rustc.exe', 'link.exe', 'cl.exe', 'git.exe', 'node.exe', 'python.exe'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    }
    $choco = 'C:\ProgramData\chocolatey\bin\choco.exe'
    & $choco feature enable -n allowGlobalConfirmation
    & $choco install git --params '/GitOnlyOnPath /NoAutoCrlf' --no-progress
    & $choco install python313 pwsh nodejs-lts strawberryperl ninja 7zip gh llvm jq --no-progress
    & $choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' --no-progress
    & $choco install visualstudio2022buildtools --no-progress --package-parameters `
        '--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.ATL --includeRecommended --passive --locale en-US'
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { throw "Visual Studio Build Tools install failed ($LASTEXITCODE)" }
    # GitHub's images put Git's bash on PATH for `shell: bash`.
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    foreach ($dir in @('E:\ci-cache\cargo\bin', 'C:\Program Files\Git\bin', 'C:\Python313', 'C:\Python313\Scripts', 'C:\Program Files\LLVM\bin')) {
        if ($machinePath -notlike "*$dir*") { $machinePath = "$machinePath;$dir" }
    }
    [Environment]::SetEnvironmentVariable('Path', $machinePath, 'Machine')
    [Environment]::SetEnvironmentVariable('LIBCLANG_PATH', 'C:\Program Files\LLVM\bin', 'Machine')
    & 'C:\Program Files\Git\bin\git.exe' config --system core.longpaths true
    # Ops Agent: disk, memory and log metrics for the disk alert.
    $agent = "$env:TEMP\add-google-cloud-ops-agent-repo.ps1"
    Invoke-WebRequest -UseBasicParsing 'https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.ps1' -OutFile $agent
    & $agent -AlsoInstall
    New-Item -ItemType File -Force -Path "$state\packages-v1" | Out-Null
}
# Operator access: OpenSSH through IAP (`gcloud compute ssh ci-windows-x64 --tunnel-through-iap`),
# with keys from metadata via the guest agent (instance metadata enable-windows-ssh=TRUE).
if (-not (Get-Service sshd -ErrorAction SilentlyContinue)) {
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
    & googet -noconfirm=true install google-compute-engine-ssh 2>&1 | Out-Host
    Set-Service sshd -StartupType Automatic
}
Start-Service sshd -ErrorAction SilentlyContinue

# Java 21 for runmat-java and the AOT Java adapter tests (GitHub's images ship a JDK).
if (-not (Test-Path "$state\jdk21-v1")) {
    & 'C:\ProgramData\chocolatey\bin\choco.exe' install temurin21 --no-progress
    if ($LASTEXITCODE -ne 0) { throw "Temurin 21 install failed ($LASTEXITCODE)" }
    New-Item -ItemType File -Force -Path "$state\jdk21-v1" | Out-Null
}

# Desktop release signing (private repo): Azure Artifact Signing runs signtool with
# Azure.CodeSigning.Dlib.dll, a mixed-mode .NET 8 library, and signs in through the
# Azure CLI. Without the .NET 8 runtime signtool aborts (exit code 3, no message).
# GitHub's images ship both.
if (-not (Test-Path "$state\signing-v1")) {
    & 'C:\ProgramData\chocolatey\bin\choco.exe' install dotnet-8.0-runtime azure-cli --no-progress
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010) { throw ".NET 8 runtime / Azure CLI install failed ($LASTEXITCODE)" }
    New-Item -ItemType File -Force -Path "$state\signing-v1" | Out-Null
}

# Jobs find the shared rustup proxies on PATH.
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
if ($machinePath -notlike '*E:\ci-cache\cargo\bin*') {
    [Environment]::SetEnvironmentVariable('Path', "E:\ci-cache\cargo\bin;$machinePath", 'Machine')
}
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine')

# ---- runner account: a local administrator (jobs install tools with choco).
# Its password is random, set here, used once for the services and never stored.
if (-not (Get-LocalUser -Name runner -ErrorAction SilentlyContinue)) {
    New-LocalUser -Name runner -NoPassword -AccountNeverExpires -UserMayNotChangePassword -ErrorAction Stop | Out-Null
    Add-LocalGroupMember -Group Administrators -Member runner
}
$bytes = New-Object byte[] 36; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$password = [Convert]::ToBase64String($bytes) + 'aA1!'
Set-LocalUser -Name runner -Password (ConvertTo-SecureString $password -AsPlainText -Force) -PasswordNeverExpires $true -ErrorAction Stop
# The reset makes the account's DPAPI-protected data unreadable; the Azure CLI's sign-in
# cache is one such file, and `az login` fails on it ("Key not valid for use in specified
# state"). Drop it so the next sign-in starts clean.
Remove-Item -Recurse -Force 'C:\Users\runner\.azure' -ErrorAction SilentlyContinue

# ---- cache layout and config
foreach ($dir in 'lanes', 'locks', 'cargo', 'rustup', 'toolcache', 'npm', 'pip') { New-Item -ItemType Directory -Force -Path "E:\ci-cache\$dir" | Out-Null }
$cacheTotal = [math]::Floor((Get-Volume -DriveLetter E).Size / 1GB)
@{
    work = 'D:\ci-work'; cache = 'E:\ci-cache'; state = $state; runner_root = 'C:\ci-runner'; system_root = 'C:\'
    cache_cap_gib = [math]::Floor($cacheTotal * 0.80); cache_floor_gib = [math]::Floor($cacheTotal * 0.12)
    lane_cap_gib = 200; lane_max_idle_days = 14; work_floor_gib = 120; system_floor_gib = 25; idle_minutes = [int]$idleMinutes
} | ConvertTo-Json | Set-Content -Encoding ascii 'C:\ci\ci-runner.json'

$python = 'C:\Python313\python.exe'

# ---- Rust: one rustup home per slot, because `rustup default` is global per home
$env:CARGO_HOME = 'E:\ci-cache\cargo'
if (-not (Test-Path 'E:\ci-cache\cargo\bin\rustup.exe')) {
    Invoke-WebRequest -UseBasicParsing 'https://win.rustup.rs/x86_64' -OutFile "$env:TEMP\rustup-init.exe"
    $env:RUSTUP_HOME = 'E:\ci-cache\rustup\bootstrap'
    & "$env:TEMP\rustup-init.exe" -y --no-modify-path --default-toolchain none --profile minimal 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "rustup-init failed ($LASTEXITCODE)" }
}

# ---- runner binaries
$dist = "C:\actions-runner-dist\$runnerVersion"
if (-not (Test-Path "$dist\.complete")) {
    # A reset during extraction leaves a partial tree; the marker is written last.
    if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
    New-Item -ItemType Directory -Force -Path $dist | Out-Null
    $zip = "$env:TEMP\actions-runner.zip"
    Invoke-WebRequest -UseBasicParsing "https://github.com/actions/runner/releases/download/v$runnerVersion/actions-runner-win-x64-$runnerVersion.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dist -Force -ErrorAction Stop
    New-Item -ItemType File -Path "$dist\.complete" | Out-Null
}

$registration = $null
$registrationText = Get-Md 'ci-registration'
if ($registrationText) { $registration = $registrationText | ConvertFrom-Json }

foreach ($entry in $slots) {
    $slot, $repo = $entry -split '=', 2
    $dir = "C:\ci-runner\$slot"
    foreach ($sub in 'w', 'home', 'tmp', 'cold-target') { New-Item -ItemType Directory -Force -Path "D:\ci-work\$slot\$sub" | Out-Null }
    $env:RUSTUP_HOME = "E:\ci-cache\rustup\$slot"
    foreach ($toolchain in '1.90.0', '1.93.0') {
        & 'E:\ci-cache\cargo\bin\rustup.exe' toolchain install $toolchain --profile minimal -c rustfmt,clippy --no-self-update 2>&1 | Out-Null
    }
    & 'E:\ci-cache\cargo\bin\rustup.exe' default 1.90.0 2>&1 | Out-Null

    $envFile = @"
CI_SLOT=$slot
JAVA_HOME=$([Environment]::GetEnvironmentVariable('JAVA_HOME', 'Machine'))
TEMP=D:\ci-work\$slot\tmp
TMP=D:\ci-work\$slot\tmp
CARGO_HOME=E:\ci-cache\cargo
RUSTUP_HOME=E:\ci-cache\rustup\$slot
RUNMAT_CI_CACHE=E:\ci-cache
RUNMAT_CI_TOOL=C:\ci\common\ci_cache.py
RUNMAT_VCPKG_ROOT=E:\ci-cache\vcpkg-runmat
AGENT_TOOLSDIRECTORY=E:\ci-cache\toolcache
RUNNER_TOOL_CACHE=E:\ci-cache\toolcache
npm_config_cache=E:\ci-cache\npm
PIP_CACHE_DIR=E:\ci-cache\pip
ACTIONS_RUNNER_HOOK_JOB_STARTED=C:\ci\windows\hooks\job-started.ps1
ACTIONS_RUNNER_HOOK_JOB_COMPLETED=C:\ci\windows\hooks\job-completed.ps1
"@
    if (-not (Test-Path "$dir\.runner")) {
        $token = if ($registration) { $registration.$repo } else { $null }
        if (-not $token) { Write-Host "slot ${slot}: not registered and no registration token in metadata"; continue }
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        Copy-Item -Recurse $dist $dir
        Set-Content -Encoding ascii "$dir\.env" $envFile
        $runnerName = "gcp-$instance-$slot"
        & "$dir\config.cmd" --unattended --replace --url "https://github.com/$repo" --token $token `
            --name $runnerName --labels $label --work "D:\ci-work\$slot\w" --disableupdate `
            --runasservice --windowslogonaccount '.\runner' --windowslogonpassword $password | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Host "slot ${slot}: registration failed ($LASTEXITCODE)"; Remove-Item -Recurse -Force $dir; continue }
        Write-Host "slot ${slot}: registered for $repo with label $label"
    }
    Set-Content -Encoding ascii "$dir\.env" $envFile
    # The runner reads PATH for jobs from .path; put the shared rustup proxies first.
    Set-Content -Encoding ascii "$dir\.path" ("E:\ci-cache\cargo\bin;" + [Environment]::GetEnvironmentVariable('Path', 'Machine'))
    $service = Get-Service "actions.runner.*" | Where-Object { $_.Name -like "*.gcp-$instance-$slot" }
    if ($service) {
        # The password changes at every boot. Start manually, after the slot is wiped below.
        & sc.exe config $service.Name obj= '.\runner' password= $password start= demand | Out-Null
    }
}
& icacls 'D:\ci-work' /grant 'runner:(OI)(CI)F' /T /C /Q | Out-Null
& icacls 'E:\ci-cache' /grant 'runner:(OI)(CI)F' /T /C /Q | Out-Null
& icacls 'C:\ci-runner' /grant 'runner:(OI)(CI)F' /T /C /Q | Out-Null

# ---- idle shutdown (every 2 minutes) and the daily prune
$taskUser = 'NT AUTHORITY\SYSTEM'
$idle = New-ScheduledTaskAction -Execute $python -Argument 'C:\ci\common\ci_cache.py idle-check'
$every2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName 'ci-idle' -Action $idle -Trigger $every2 -User $taskUser -RunLevel Highest -Force | Out-Null
$prune = New-ScheduledTaskAction -Execute $python -Argument 'C:\ci\common\ci_cache.py prune-if-due'
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(10) -RepetitionInterval (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName 'ci-prune' -Action $prune -Trigger $hourly -User $taskUser -RunLevel Highest -Force | Out-Null

foreach ($entry in $slots) {
    $slot = ($entry -split '=', 2)[0]
    & $python C:\ci\common\ci_cache.py slot-start $slot
}
Get-Service 'actions.runner.*' | Start-Service
Write-Host 'ci-runner setup complete'
