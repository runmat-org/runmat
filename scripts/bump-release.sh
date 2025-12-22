#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/bump-release.sh <new_version> [--yes]
# Example: scripts/bump-release.sh 0.0.11 --yes

ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <new_version> [--yes]" >&2
  exit 1
fi

NEW_VERSION="$1"
YES_FLAG="${2:-}"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
  echo "Error: version must be semver (e.g., 0.0.11)" >&2
  exit 1
fi

# Ensure on main and clean
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "Error: must run on main (current: $CURRENT_BRANCH)" >&2
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

TAG="v${NEW_VERSION}"
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Error: tag ${TAG} already exists" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Error: cargo not found on PATH" >&2
  exit 1
fi

if ! cargo workspaces -V >/dev/null 2>&1; then
  echo "Installing cargo-workspaces..."
  cargo install cargo-workspaces --locked
fi

echo "Bumping workspace versions to ${NEW_VERSION}..."
cargo workspaces version custom "${NEW_VERSION}" \
  --force '*' \
  --no-git-commit \
  --exact \
  --all \
  --yes

echo "Running a quick build check..."
cargo check -q

if [ "$YES_FLAG" != "--yes" ]; then
  read -r -p "Commit and tag release ${TAG}? [y/N] " RESP
  case "$RESP" in
    [yY][eE][sS]|[yY]) ;;
    *) echo "Aborting."; exit 1;;
  esac
fi

echo "Committing version bumps..."
git add -A
git commit -m "chore(release): ${TAG}"

echo "Creating tag ${TAG}..."
git tag -a "${TAG}" -m "Release ${TAG}"

echo "Pushing main and tag to origin..."
git push origin main
git push origin "${TAG}"

echo "Done. GitHub Actions will now publish the release for ${TAG}."


