#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as the Linux runner user, not as root. It will use sudo for systemd changes." >&2
  exit 1
fi

EXPECTED_HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
if [[ -z "${EXPECTED_HOME}" ]]; then
  echo "Could not determine passwd home for $(id -un)" >&2
  exit 1
fi
export HOME="${EXPECTED_HOME}"

CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-${HOME}/actions-runner}"
RUNNER_PATH="${CARGO_HOME}/bin:/usr/sbin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games"

if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
  echo "rustup was not found at ${CARGO_HOME}/bin/rustup. Run infra/scripts/bootstrap-linux-runner.sh first." >&2
  exit 1
fi

mapfile -t RUNNER_SERVICES < <(systemctl list-unit-files 'actions.runner*.service' --no-legend 2>/dev/null | awk '{print $1}')
MATCHED_SERVICE=0
for service in "${RUNNER_SERVICES[@]}"; do
  service_user="$(systemctl show "${service}" -p User --value 2>/dev/null || true)"
  service_workdir="$(systemctl show "${service}" -p WorkingDirectory --value 2>/dev/null || true)"
  if [[ "${service_user}" != "$(id -un)" || "${service_workdir}" != "${RUNNER_WORKDIR}" ]]; then
    continue
  fi

  dropin_dir="/etc/systemd/system/${service}.d"
  dropin_file="${dropin_dir}/10-runmat-env.conf"
  echo "Configuring systemd environment override for ${service}"
  sudo mkdir -p "${dropin_dir}"
  sudo tee "${dropin_file}" >/dev/null <<EOF
[Service]
Environment=HOME=${HOME}
Environment=PATH=${RUNNER_PATH}
EOF
  MATCHED_SERVICE=1
done

if [[ "${MATCHED_SERVICE}" -eq 0 ]]; then
  echo "No matching GitHub Actions runner service found for user $(id -un) at ${RUNNER_WORKDIR}" >&2
  exit 1
fi

echo "Reloading systemd and restarting matching GitHub Actions runner services"
sudo systemctl daemon-reload
for service in "${RUNNER_SERVICES[@]}"; do
  service_user="$(systemctl show "${service}" -p User --value 2>/dev/null || true)"
  service_workdir="$(systemctl show "${service}" -p WorkingDirectory --value 2>/dev/null || true)"
  if [[ "${service_user}" == "$(id -un)" && "${service_workdir}" == "${RUNNER_WORKDIR}" ]]; then
    sudo systemctl restart "${service}"
    systemctl show "${service}" -p Environment
  fi
done

echo
echo "Linux runner service environment configuration complete."
