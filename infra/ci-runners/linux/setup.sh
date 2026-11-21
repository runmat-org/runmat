#!/usr/bin/env bash
# Provision or refresh a Linux runner VM. The GCE startup script runs this as
# root at every boot (see startup.sh); every step is idempotent.
set -euo pipefail

CI=/opt/ci
md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }

SLOTS="$(md ci-slots)"          # e.g. "pub1=runmat-org/runmat pub2=runmat-org/runmat prv1=runmat-org/runmat-private"
LABEL="$(md ci-label)"          # e.g. runmat-gcp-linux-x64
RUNNER_VERSION="$(md ci-runner-version)"
CACHE_CAP_GIB="$(md ci-cache-cap-gib)"
LANE_CAP_GIB="$(md ci-lane-cap-gib)"
IDLE_MINUTES="$(md ci-idle-minutes)"

# ---- disks: work and cache live on their own persistent disks so a full cache
# or a runaway job can never fill the boot disk.
mount_disk() {
  local device="/dev/disk/by-id/google-$1" mountpoint="$2"
  [[ -e "$device" ]] || { echo "missing disk $1" >&2; exit 1; }
  if ! blkid "$device" >/dev/null 2>&1; then
    mkfs.ext4 -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard -L "$1" "$device"
  fi
  mkdir -p "$mountpoint"
  local uuid; uuid="$(blkid -s UUID -o value "$device")"
  grep -q "$uuid" /etc/fstab || echo "UUID=$uuid $mountpoint ext4 discard,defaults,nofail,noatime 0 2" >> /etc/fstab
  mountpoint -q "$mountpoint" || mount "$mountpoint"
}
mount_disk ci-work /ci/work
mount_disk ci-cache /ci/cache

# ---- packages (first boot only; the marker keeps later boots fast)
if [[ ! -f /var/lib/ci-runner/packages-v1 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    autoconf autoconf-archive automake build-essential ca-certificates cmake curl docker.io \
    file gdb git git-lfs gnupg jq libdbus-1-dev libegl1-mesa-dev libgl1-mesa-dev libhdf5-dev \
    liblapack-dev libopenblas-dev libssl-dev libtool libudev-dev libvulkan1 mesa-vulkan-drivers libwayland-dev \
    libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxkbcommon-dev libxkbcommon-x11-0 libxrandr-dev \
    libzmq3-dev ninja-build pkg-config python3 python3-pip python3-venv unzip xauth xvfb zip zstd \
    libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav patchelf rsync tar
  # GitHub CLI
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /etc/apt/keyrings/githubcli.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/githubcli.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
  apt-get update && apt-get install -y gh
  # Node.js (scripts call `node` without setup-node, as on GitHub's images)
  curl -fsSL https://nodejs.org/dist/v22.21.1/node-v22.21.1-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
  # Ops Agent: disk, memory and log metrics for the disk alert
  curl -fsSL https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh | bash -s -- --also-install
  mkdir -p /var/lib/ci-runner && touch /var/lib/ci-runner/packages-v1
fi

# setup-python's prebuilt interpreters record /opt/hostedtoolcache as their
# prefix (sysconfig LIBDIR), so the tool cache must be reachable at that path.
mkdir -p /ci/cache/toolcache && chown runner:runner /ci/cache/toolcache
[[ -L /opt/hostedtoolcache ]] || { rm -rf /opt/hostedtoolcache; ln -s /ci/cache/toolcache /opt/hostedtoolcache; }

# Java 21 for runmat-java and the AOT Java adapter tests (GitHub's images ship a JDK).
if [[ ! -d /usr/lib/jvm/java-21-openjdk-amd64 ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jdk-headless
fi

# Concurrent jobs all run `sudo apt-get update/install`. Serialize them, and
# keep Ubuntu's background apt jobs from holding the lock.
cat > /usr/local/sbin/apt-get <<'WRAP'
#!/bin/sh
# Installed by infra/ci-runners: one apt-get at a time across runner slots.
exec flock /var/lock/ci-apt.lock /usr/bin/apt-get "$@"
WRAP
chmod 0755 /usr/local/sbin/apt-get
cp /usr/local/sbin/apt-get /usr/local/bin/apt-get
echo 'DPkg::Lock::Timeout "900";' > /etc/apt/apt.conf.d/90ci-runner-lock
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service >/dev/null 2>&1 || true

id runner >/dev/null 2>&1 || useradd --create-home --shell /bin/bash runner
usermod -aG docker runner
echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner && chmod 0440 /etc/sudoers.d/runner
# Docker images and build caches live on the cache disk, never the boot disk; the daily prune trims them.
mkdir -p /etc/docker /ci/cache/docker
if ! grep -q '/ci/cache/docker' /etc/docker/daemon.json 2>/dev/null; then
  echo '{"data-root": "/ci/cache/docker", "log-driver": "local"}' > /etc/docker/daemon.json
  systemctl restart docker || true
fi
systemctl enable --now docker

# ---- cache layout (owned by the runner user)
mkdir -p /ci/cache/{lanes,locks,cargo,rustup,toolcache,npm,pip} /var/lib/ci-runner
chown runner:runner /ci/cache /ci/cache/{lanes,locks,cargo,rustup,toolcache,npm,pip}

cache_total_gib=$(( $(df --output=size -B1G /ci/cache | tail -1) ))
cat > /etc/ci-runner.json <<JSON
{
  "work": "/ci/work",
  "cache": "/ci/cache",
  "state": "/var/lib/ci-runner",
  "runner_root": "/opt/ci-runner",
  "system_root": "/",
  "cache_cap_gib": ${CACHE_CAP_GIB:-$(( cache_total_gib * 80 / 100 ))},
  "cache_floor_gib": $(( cache_total_gib * 12 / 100 )),
  "lane_cap_gib": ${LANE_CAP_GIB:-160},
  "lane_max_idle_days": 14,
  "work_floor_gib": 100,
  "system_floor_gib": 15,
  "idle_minutes": ${IDLE_MINUTES:-30}
}
JSON

# ---- Rust toolchains: one rustup home per slot, because `rustup default` is global per home
install_rustup() {
  local home="$1"
  if [[ ! -x /ci/cache/cargo/bin/rustup ]]; then
    curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh
    sudo -u runner env CARGO_HOME=/ci/cache/cargo RUSTUP_HOME="$home" sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain none
  fi
  for toolchain in 1.90.0 1.93.0; do
    sudo -u runner env CARGO_HOME=/ci/cache/cargo RUSTUP_HOME="$home" /ci/cache/cargo/bin/rustup toolchain install "$toolchain" --profile minimal -c rustfmt,clippy --no-self-update >/dev/null
  done
  sudo -u runner env CARGO_HOME=/ci/cache/cargo RUSTUP_HOME="$home" /ci/cache/cargo/bin/rustup default 1.90.0 >/dev/null
}

# ---- runner binaries
dist=/opt/actions-runner-dist/$RUNNER_VERSION
if [[ ! -x "$dist/run.sh" ]]; then
  mkdir -p "$dist"
  curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" | tar -xz -C "$dist"
  "$dist/bin/installdependencies.sh" >/dev/null
fi

install -d /etc/systemd/system
install -m 0644 "$CI"/linux/systemd/* /etc/systemd/system/
systemctl daemon-reload

# Jobs get HOME=/ci/work/<slot>/home, which every job hook wipes. The runner
# user's passwd home stays /home/runner; rustup-init compares the two and
# RUSTUP_INIT_SKIP_SUDO_CHECK (in each slot's .env) tells it the difference is
# intended. Tools that read the passwd home directly (ssh) must not keep state
# there between jobs, so root owns it and the runner user cannot write to it.
chown root:runner /home/runner && chmod 0550 /home/runner
# No job runs during boot: a provisioning lock left by a reset belongs to no one.
rmdir /ci/cache/vcpkg-runmat.lock 2>/dev/null || true

registration="$(md ci-registration)"   # {"owner/repo": "<registration token>"}; short-lived, removed by the operator
for entry in $SLOTS; do
  slot="${entry%%=*}" repo="${entry#*=}"
  dir=/opt/ci-runner/$slot
  mkdir -p "/ci/work/$slot"/{w,home,tmp,cold-target}
  chown -R runner:runner "/ci/work/$slot"
  install_rustup "/ci/cache/rustup/$slot"
  if [[ ! -f "$dir/.runner" ]]; then
    token="$(jq -r --arg repo "$repo" '.[$repo] // empty' <<<"$registration" 2>/dev/null || true)"
    if [[ -z "$token" ]]; then echo "slot $slot: not registered and no registration token in metadata"; continue; fi
    rm -rf "$dir"; mkdir -p "$dir"; cp -a "$dist/." "$dir/"; chown -R runner:runner "$dir"
    sudo -u runner env HOME="/ci/work/$slot/home" "$dir/config.sh" --unattended --replace \
      --url "https://github.com/$repo" --token "$token" --name "gcp-$(hostname -s)-$slot" \
      --labels "$LABEL" --work "/ci/work/$slot/w" --disableupdate >/dev/null
    echo "slot $slot: registered for $repo with label $LABEL"
  fi
  cat > "$dir/.env" <<ENV
CI_SLOT=$slot
HOME=/ci/work/$slot/home
TMPDIR=/ci/work/$slot/tmp
CARGO_HOME=/ci/cache/cargo
RUSTUP_HOME=/ci/cache/rustup/$slot
RUSTUP_INIT_SKIP_SUDO_CHECK=yes
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
RUNMAT_CI_CACHE=/ci/cache
RUNMAT_CI_TOOL=/opt/ci/common/ci_cache.py
RUNMAT_VCPKG_ROOT=/ci/cache/vcpkg-runmat
AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache
RUNNER_TOOL_CACHE=/opt/hostedtoolcache
npm_config_cache=/ci/cache/npm
PIP_CACHE_DIR=/ci/cache/pip
ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/ci/linux/hooks/job-started.sh
ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/opt/ci/linux/hooks/job-completed.sh
LANG=C.UTF-8
ENV
  echo "/ci/cache/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" > "$dir/.path"
  chown runner:runner "$dir/.env" "$dir/.path"
  systemctl enable "ci-runner@$slot.service" >/dev/null 2>&1
done

python3 "$CI/common/ci_cache.py" prune-if-due || true
systemctl enable --now ci-idle.timer ci-prune.timer
systemctl enable ci-runners.target
systemctl start ci-runners.target
for entry in $SLOTS; do systemctl start "ci-runner@${entry%%=*}.service" || true; done
echo "ci-runner setup complete"
