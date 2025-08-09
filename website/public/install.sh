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

# Functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Banner
echo -e "${BLUE}"
cat << 'EOF'
 ____            _   __  __       _   
|  _ \ _   _ ___| |_|  \/  | __ _| |_ 
| |_) | | | / __| __| |\/| |/ _` | __|
|  _ <| |_| \__ \ |_| |  | | (_| | |_ 
|_| \_\\__,_|___/\__|_|  |_|\__,_|\__|

High-performance MATLAB/Octave runtime
EOF
echo -e "${NC}"

log "Starting RunMat installation..."

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

log "Detected OS: $OS"
log "Detected architecture: $ARCH"

case $OS in
    linux)
        case $ARCH in
            x86_64) PLATFORM="Linux-x86_64" ;;
            aarch64|arm64) PLATFORM="Linux-aarch64" ;;
            *) error "Unsupported architecture: $ARCH" ;;
        esac
        ;;
    darwin)
        case $ARCH in
            x86_64) PLATFORM="Darwin-x86_64" ;;
            arm64) PLATFORM="Darwin-aarch64" ;;
            *) error "Unsupported architecture: $ARCH" ;;
        esac
        ;;
    *)
        error "Unsupported OS: $OS"
        ;;
esac

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

# Download and install
DOWNLOAD_URL="https://github.com/$REPO/releases/download/$LATEST_RELEASE/runmat-$PLATFORM.tar.gz"
TEMP_DIR=$(mktemp -d)

log "Downloading from: $DOWNLOAD_URL"
if ! curl -L "$DOWNLOAD_URL" | tar -xz -C "$TEMP_DIR"; then
    error "Failed to download or extract RunMat"
fi

# Create install directory
log "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Install binary
log "Installing RunMat binary..."
if ! cp "$TEMP_DIR/$BINARY_NAME" "$INSTALL_DIR/"; then
    error "Failed to install binary"
fi

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

    # Apply for current session
    export PATH="$INSTALL_DIR:$PATH"
    warn "Added $INSTALL_DIR to your PATH. Restart your terminal or 'source' your shell profile to take effect."
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