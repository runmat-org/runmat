#!/usr/bin/env bash
# GCE startup script: unpack the setup bundle from metadata and run setup.sh.
set -euo pipefail
bundle="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/ci-bundle)"
rm -rf /opt/ci.new && mkdir -p /opt/ci.new
base64 -d <<<"$bundle" | tar -xz -C /opt/ci.new
rm -rf /opt/ci && mv /opt/ci.new /opt/ci
exec bash /opt/ci/linux/setup.sh
