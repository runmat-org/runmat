#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <macos-binary>" >&2
  exit 2
fi

binary="$1"
if [[ ! -f "$binary" ]]; then
  echo "macOS binary does not exist: $binary" >&2
  exit 1
fi
binary="$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")"

required_environment=(
  MACOS_CERT_P12
  MACOS_CERT_PASSWORD
  MACOS_CERT_IDENTITY
  APPLE_NOTARIZE_KEY_ID
  APPLE_NOTARIZE_ISSUER_ID
  APPLE_NOTARIZE_PRIVATE_KEY
)
for variable in "${required_environment[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "$variable is required to sign and notarize macOS artifacts" >&2
    exit 1
  fi
done

work_directory="$(mktemp -d "${TMPDIR:-/tmp}/runmat-macos-signing.XXXXXX")"
keychain="$work_directory/build.keychain-db"
keychain_password="$(uuidgen)"
certificate="$work_directory/certificate.p12"
notary_key="$work_directory/notary-key.p8"
notary_archive="$work_directory/notarize-submit.zip"
entitlements="$work_directory/entitlements.plist"

cleanup() {
  security delete-keychain "$keychain" >/dev/null 2>&1 || true
  rm -rf "$work_directory"
}
trap cleanup EXIT

printf '%s' "$MACOS_CERT_P12" | base64 --decode > "$certificate"
security create-keychain -p "$keychain_password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$keychain_password" "$keychain"
security import "$certificate" -k "$keychain" -P "$MACOS_CERT_PASSWORD" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple: -s -k "$keychain_password" "$keychain"

cat > "$entitlements" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.allow-jit</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
</dict>
</plist>
EOF

codesign --force --timestamp --options runtime --entitlements "$entitlements" \
  --keychain "$keychain" --sign "$MACOS_CERT_IDENTITY" "$binary"
codesign --verify --strict --verbose=2 "$binary"

printf '%s' "$APPLE_NOTARIZE_PRIVATE_KEY" | base64 --decode > "$notary_key"
/usr/bin/ditto -c -k --keepParent "$binary" "$notary_archive"
xcrun notarytool submit "$notary_archive" \
  --key "$notary_key" \
  --key-id "$APPLE_NOTARIZE_KEY_ID" \
  --issuer "$APPLE_NOTARIZE_ISSUER_ID" \
  --wait

# Notarization tickets cannot be stapled to a bare executable. Gatekeeper uses
# Apple's online ticket for the signed CLI, so assess the exact binary that the
# workflow will package.
spctl --assess --type execute --verbose=4 "$binary"
