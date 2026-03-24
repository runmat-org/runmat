#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/cut-release.sh <version>
# Example: scripts/cut-release.sh 0.3.1

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
TAG="v${VERSION}"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
  echo "Error: version must be semver (e.g. 0.3.1)" >&2
  exit 1
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
  echo "Error: must run on main" >&2
  exit 1
fi

#if [ -n "$(git status --porcelain)" ]; then
#  echo "Error: working tree not clean. Commit or stash changes first." >&2
#  exit 1
#fi

git fetch origin +refs/heads/main:refs/remotes/origin/main
if ! git merge-base --is-ancestor HEAD refs/remotes/origin/main; then
  echo "Error: local main is behind origin/main. Pull/rebase first." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Error: tag ${TAG} already exists" >&2
  exit 1
fi

if ! cargo workspaces -V >/dev/null 2>&1; then
  echo "Installing cargo-workspaces..."
  cargo install cargo-workspaces --locked
fi

echo "Setting crate versions to ${VERSION}..."
cargo workspaces version custom "${VERSION}" \
  --force '*' \
  --no-git-commit \
  --exact \
  --all \
  --yes

echo "Updating bindings/ts package version to ${VERSION}..."
(
  cd bindings/ts
  npm version "$VERSION" --no-git-tag-version
)

echo "Running a quick build check..."
cargo check -q

git add Cargo.toml Cargo.lock crates bindings/ts/package.json bindings/ts/package-lock.json
git commit -m "chore(release): ${TAG}"

git push origin main
git tag -a "${TAG}" -m "Release ${TAG}"
git push origin "${TAG}"

echo "Created ${TAG}. GitHub Actions will build binaries, create the GitHub release, then publish crates.io and npm artifacts."
