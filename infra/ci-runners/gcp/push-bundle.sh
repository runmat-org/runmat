#!/usr/bin/env bash
# Publish infra/ci-runners to a runner VM's metadata and rerun its setup.
# Usage: gcp/push-bundle.sh linux|windows [--register]
#   --register  also hands the VM short-lived registration tokens for slots
#               that are not registered yet (requires repo admin in gh).
# Tokens go into instance metadata only for the setup run and are removed after.
set -euo pipefail
os="$1"; register="${2:-}"
PROJECT=runmat ZONE=us-central1-b
here="$(cd "$(dirname "$0")/.." && pwd)"
vm="ci-${os}-x64"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
(cd "$here" && COPYFILE_DISABLE=1 tar --no-xattrs -czf - common "$os") | base64 > "$work/bundle"
gcloud compute instances add-metadata "$vm" --project "$PROJECT" --zone "$ZONE" --metadata-from-file "ci-bundle=$work/bundle" >/dev/null
if [[ "$register" == "--register" ]]; then
  umask 077
  python3 - > "$work/reg.json" <<'PY'
import json, subprocess
token = lambda repo: subprocess.check_output(["gh", "api", "-X", "POST", f"repos/{repo}/actions/runners/registration-token", "-q", ".token"], text=True).strip()
print(json.dumps({repo: token(repo) for repo in ["runmat-org/runmat", "runmat-org/runmat-private"]}))
PY
  gcloud compute instances add-metadata "$vm" --project "$PROJECT" --zone "$ZONE" --metadata-from-file "ci-registration=$work/reg.json" >/dev/null
fi
if [[ "$os" == linux ]]; then
  gcloud compute ssh "$vm" --project "$PROJECT" --zone "$ZONE" --tunnel-through-iap --command 'sudo google_metadata_script_runner startup' 2>&1 | tail -3
else
  # Windows reruns its startup script on reset; setup.ps1 is idempotent.
  gcloud compute instances reset "$vm" --project "$PROJECT" --zone "$ZONE" >/dev/null
  echo "reset $vm; setup runs at boot (see C:\\ci\\setup.log or the serial console)"
fi
if [[ "$register" == "--register" && "$os" == linux ]]; then
  gcloud compute instances remove-metadata "$vm" --project "$PROJECT" --zone "$ZONE" --keys ci-registration >/dev/null
fi
