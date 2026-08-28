#!/usr/bin/env bash
set -euo pipefail

# Validate a reviewed release version on main, then tag that exact commit.
# Usage: scripts/cut-release.sh <version> [--dry-run]
# Example: scripts/cut-release.sh 0.6.2

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <version> [--dry-run]" >&2
  exit 1
fi

VERSION="$1"
DRY_RUN="false"
if [ $# -eq 2 ]; then
  if [ "$2" != "--dry-run" ]; then
    echo "Error: unsupported option: $2" >&2
    exit 1
  fi
  DRY_RUN="true"
fi
TAG="v${VERSION}"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
  echo "Error: version must be semver (e.g. 0.6.2)" >&2
  exit 1
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  echo "Error: must run on main" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Error: working tree not clean. Commit or stash changes first." >&2
  exit 1
fi

git fetch origin +refs/heads/main:refs/remotes/origin/main --tags
if [ "$(git rev-parse HEAD)" != "$(git rev-parse refs/remotes/origin/main)" ]; then
  echo "Error: local main must exactly match origin/main." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Error: tag ${TAG} already exists" >&2
  exit 1
fi

echo "Verifying reviewed release state for ${VERSION}..."
node scripts/verify-release-version.mjs "$VERSION"
cargo check -q --locked

if [ "$DRY_RUN" = "true" ]; then
  echo "Release state is valid. Dry run complete; ${TAG} was not created."
  exit 0
fi

git tag -a "${TAG}" -m "Release ${TAG}"
git push origin "${TAG}"

echo "Created ${TAG}. GitHub Actions will build binaries, create the GitHub release, then publish crates.io and npm artifacts."
