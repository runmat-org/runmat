#!/bin/bash
# RunMat Installation Script
# Usage: curl -fsSL https://runmat.org/install.sh | sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Constants
REPO="runmat-org/runmat"
BINARY_NAME="runmat"
INSTALL_DIR="$HOME/.local/bin"
WEBSITE_URL="https://runmat.org"

# Optional: Telemetry relay (best-effort, anonymous)
TELEMETRY_ID_FILE="$HOME/.runmat/telemetry_id"
TELEMETRY_ENDPOINT="https://runmat.org/api/telemetry"

log() {
    printf "%b[INFO]%b %s\n" "$GREEN" "$NC" "$1"
}

warn() {
    printf "%b[WARN]%b %s\n" "$YELLOW" "$NC" "$1"
}

error() {
    printf "%b[ERROR]%b %s\n" "$RED" "$NC" "$1"
    exit 1
}

# Generate or read a stable anonymous client id
_telemetry_client_id() {
    if [ -f "$TELEMETRY_ID_FILE" ]; then
        cat "$TELEMETRY_ID_FILE" 2>/dev/null || true
        return
    fi
    local cid
    if command -v uuidgen >/dev/null 2>&1; then
        cid="$(uuidgen)"
    elif command -v python3 >/dev/null 2>&1; then
        cid="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
    else
        cid="$(date +%s)-$RANDOM"
    fi
    mkdir -p "$(dirname "$TELEMETRY_ID_FILE")" 2>/dev/null || true
    echo "$cid" > "$TELEMETRY_ID_FILE" 2>/dev/null || true
    echo "$cid"
}

# Send anonymous telemetry event (non-blocking)
_send_telemetry() {
    # usage: _send_telemetry event_name
    local EVENT_NAME="$1"
    local CID
    CID="$(_telemetry_client_id)"
    curl -s -m 3 -o /dev/null -X POST -H "Content-Type: application/json" \
      --data "{\"event\":\"$EVENT_NAME\",\"os\":\"$OS\",\"arch\":\"$ARCH\",\"platform\":\"$PLATFORM\",\"release\":\"${LATEST_RELEASE:-unknown}\",\"method\":\"shell\",\"cid\":\"$CID\"}" \
      "$TELEMETRY_ENDPOINT" >/dev/null 2>&1 || true
}

# Trap failures so we can emit a failure event without interrupting output
trap '_send_telemetry install_failed' ERR

# Banner
printf "%b" "$BLUE"
cat << 'EOF'
  _____                 __  __       _   
 |  __ \               |  \/  |     | |  
 | |__) | _   _ _ __   | \  / | __ _| |_ 
 |  _  / | | | | '_ \  | |\/| |/ _` | __|
 | | \ \ |_| | | | | | | |  | | (_| | |_ 
 |_|  \_\__,_|_| |_| |_|_|  |_|\__,_|\__|
EOF
printf "%b\n" "$NC"

log "Starting RunMat installation..."

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

log "Detected OS: $OS"
log "Detected architecture: $ARCH"

# Map to release artifact naming (new scheme) and keep legacy fallback
case $OS in
    linux)
        case $ARCH in
            x86_64)
                PLATFORM="linux-x86_64"
                ;;
            aarch64|arm64)
                warn "Linux ARM64 builds are not yet available."
                warn "Please build from source (requires Rust toolchain) or use a container image."
                error "Unsupported architecture for prebuilt binaries: $ARCH"
                ;;
            *)
                error "Unsupported architecture: $ARCH"
                ;;
        esac
        ;;
    darwin)
        case $ARCH in
            x86_64) PLATFORM="macos-x86_64" ;;
            arm64) PLATFORM="macos-aarch64" ;;
            *) error "Unsupported architecture: $ARCH" ;;
        esac
        ;;
    *)
        error "Unsupported OS: $OS"
        ;;
esac

# Emit start event now that we know OS/ARCH/PLATFORM
_send_telemetry install_start

log "Installing for platform: $PLATFORM"

# Check for required tools
if ! command -v curl >/dev/null 2>&1; then
    error "curl is required but not installed. Please install curl and try again."
fi

if ! command -v tar >/dev/null 2>&1; then
    error "tar is required but not installed. Please install tar and try again."
fi

# Get latest release
log "Fetching latest release information..."
LATEST_RELEASE=$(curl -s --fail "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' 2>/dev/null)

if [ -z "$LATEST_RELEASE" ]; then
    warn "Failed to get latest release information from GitHub API, using fallback"
    LATEST_RELEASE="v0.0.1"
fi

log "Latest release: $LATEST_RELEASE"

# Download and install (try versioned and non-versioned artifact names)
TEMP_DIR=$(mktemp -d)

DOWNLOAD_BASE="https://github.com/$REPO/releases/download/$LATEST_RELEASE"
URL_CANDIDATES=(
  "$DOWNLOAD_BASE/runmat-$LATEST_RELEASE-$PLATFORM.tar.gz"
  "$DOWNLOAD_BASE/runmat-$PLATFORM.tar.gz"
)

success=""
for DOWNLOAD_URL in "${URL_CANDIDATES[@]}"; do
    log "Downloading from: $DOWNLOAD_URL"
    if curl -fL "$DOWNLOAD_URL" | tar -xz -C "$TEMP_DIR"; then
        success="yes"
        break
    fi
done

if [ -z "$success" ]; then
    error "Failed to download or extract RunMat (tried: ${URL_CANDIDATES[*]})"
fi

# Create install directory
log "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Install binary (and preserve any bundled shared libs in TEMP_DIR if present)
log "Installing RunMat binary..."
if ! cp "$TEMP_DIR/$BINARY_NAME" "$INSTALL_DIR/"; then
    error "Failed to install binary"
fi

# Copy bundled shared libraries next to the binary (non-invasive; private dir)
for lib in "$TEMP_DIR"/*.so "$TEMP_DIR"/*.dylib; do
    if [ -f "$lib" ]; then
        cp -f "$lib" "$INSTALL_DIR/" 2>/dev/null || true
    fi
done

chmod +x "$INSTALL_DIR/$BINARY_NAME"

# Cleanup
rm -rf "$TEMP_DIR"

log "RunMat installed successfully to $INSTALL_DIR/$BINARY_NAME"

# Ensure $INSTALL_DIR is in PATH (persistently and for current session)
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    SHELL_NAME=$(basename "${SHELL:-sh}")
    declare -a PROFILE_FILES
    case "$SHELL_NAME" in
        bash)
            PROFILE_FILES=("$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile")
            ;;
        zsh)
            PROFILE_FILES=("$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.profile")
            ;;
        fish)
            # Fish shell uses a different mechanism; append to config.fish
            FISH_CONFIG="$HOME/.config/fish/config.fish"
            mkdir -p "$(dirname "$FISH_CONFIG")"
            if ! grep -q "$INSTALL_DIR" "$FISH_CONFIG" 2>/dev/null; then
                echo "# Added by RunMat installer" >> "$FISH_CONFIG"
                echo "set -Ux fish_user_paths $INSTALL_DIR \$fish_user_paths" >> "$FISH_CONFIG"
                log "Added $INSTALL_DIR to PATH in $FISH_CONFIG"
            fi
            ;;
        *)
            PROFILE_FILES=("$HOME/.profile")
            ;;
    esac

    if [ "$SHELL_NAME" != "fish" ]; then
        added=false
        for file in "${PROFILE_FILES[@]}"; do
            if [ -f "$file" ]; then
                if ! grep -q "$INSTALL_DIR" "$file"; then
                    {
                        echo ""
                        echo "# Added by RunMat installer"
                        echo "export PATH=\"$INSTALL_DIR:\$PATH\""
                    } >> "$file"
                    log "Added $INSTALL_DIR to PATH in $file"
                fi
                added=true
                break
            fi
        done
        if [ "$added" = false ]; then
            # Create primary profile file if none existed
            target_file="${PROFILE_FILES[0]}"
            {
                echo "# Created by RunMat installer"
                echo "export PATH=\"$INSTALL_DIR:\$PATH\""
            } >> "$target_file"
            log "Created $target_file and added $INSTALL_DIR to PATH"
        fi
    fi

    # Apply for current session (affects this subshell only)
    export PATH="$INSTALL_DIR:$PATH"
    warn "Added $INSTALL_DIR to your PATH."
    # If interactive and not in CI, start a new login shell so PATH is live immediately
    if [ -t 1 ] && [ -n "${SHELL:-}" ] && [ -z "${CI:-}" ] && [ -z "${GITHUB_ACTIONS:-}" ] && [ -z "${RUNMAT_NO_SHELL_RELOAD:-}" ]; then
        log "Refreshing your shell to pick up PATH changes (run 'exit' to return)."
        exec "$SHELL" -l
    else
        warn "Restart your terminal or 'source' your shell profile to take effect."
    fi
else
    log "RunMat is already in your PATH"
fi

# Test installation
if command -v runmat >/dev/null 2>&1; then
    INSTALLED_VERSION=$(runmat --version 2>/dev/null || echo "unknown")
    log "Installation verified! Version: $INSTALLED_VERSION"
else
    if [ -f "$INSTALL_DIR/runmat" ]; then
        INSTALLED_VERSION=$("$INSTALL_DIR/runmat" --version 2>/dev/null || echo "unknown")
        log "Installation verified! Version: $INSTALLED_VERSION"
        warn "Run 'export PATH=\"$INSTALL_DIR:\$PATH\"' to use runmat from anywhere"
    else
        error "Installation verification failed"
    fi
fi

# Emit completion event
_send_telemetry install_complete

echo
log "Installation complete! 🎉"
echo
echo "Next steps:"
echo "  1. Start the interactive REPL: runmat"
echo "  2. Run a script: runmat run script.m"
echo "  3. Install Jupyter kernel: runmat --install-kernel"
echo "  4. Get help: runmat --help"
echo
echo "Documentation: $WEBSITE_URL/docs"
echo "Examples: $WEBSITE_URL/docs/examples"
echo
log "Happy computing with RunMat!"