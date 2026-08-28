#!/usr/bin/env bash
set -euo pipefail

# Prepare reviewed release-version changes on dev. This script does not commit
# or push; inspect the resulting diff and merge it through the normal process.
# Usage: scripts/prepare-release.sh <version>
# Example: scripts/prepare-release.sh 0.6.2

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
  echo "Error: version must be semver (e.g. 0.6.2)" >&2
  exit 1
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "dev" ]; then
  echo "Error: must run on dev" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working tree not clean. Commit or stash changes first." >&2
  exit 1
fi

git fetch origin +refs/heads/dev:refs/remotes/origin/dev
if [ "$(git rev-parse HEAD)" != "$(git rev-parse refs/remotes/origin/dev)" ]; then
  echo "Error: local dev must exactly match origin/dev." >&2
  exit 1
fi

if ! cargo workspaces -V >/dev/null 2>&1; then
  echo "Installing cargo-workspaces..."
  cargo install cargo-workspaces --locked
fi

echo "Setting workspace crate versions to ${VERSION}..."
cargo workspaces version custom "${VERSION}" \
  --force '*' \
  --no-git-commit \
  --exact \
  --all \
  --yes

echo "Setting bindings/ts package version to ${VERSION}..."
(
  cd bindings/ts
  npm version "$VERSION" --no-git-tag-version --allow-same-version
)

node scripts/verify-release-version.mjs "$VERSION"
cargo check -q --locked

echo "Release ${VERSION} is prepared. Review and commit the version changes on dev; this script did not commit or push."
