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
original_keychains=()
while IFS= read -r listed_keychain; do
  if [[ -n "$listed_keychain" ]]; then
    original_keychains+=("$listed_keychain")
  fi
done < <(
  security list-keychains -d user |
    sed -e 's/^[[:space:]]*"//' -e 's/"[[:space:]]*$//'
)

cleanup() {
  security list-keychains -d user -s "${original_keychains[@]}" >/dev/null 2>&1 || true
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
security list-keychains -d user -s "$keychain" "${original_keychains[@]}"

identity_listing="$(security find-identity -v -p codesigning "$keychain")"
signing_identities="$(
  printf '%s\n' "$identity_listing" |
    awk '$2 ~ /^[[:xdigit:]]+$/ && length($2) == 40 { print $2 }'
)"
signing_identity_count="$(printf '%s\n' "$signing_identities" | awk 'NF { count += 1 } END { print count + 0 }')"
if [[ "$signing_identity_count" -ne 1 ]]; then
  echo "expected exactly one valid code-signing identity in the imported certificate, found $signing_identity_count" >&2
  printf '%s\n' "$identity_listing" >&2
  exit 1
fi
signing_identity="$signing_identities"

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
  --keychain "$keychain" --sign "$signing_identity" "$binary"
codesign --verify --strict --verbose=2 "$binary"

printf '%s' "$APPLE_NOTARIZE_PRIVATE_KEY" | base64 --decode > "$notary_key"
/usr/bin/ditto -c -k --keepParent "$binary" "$notary_archive"
xcrun notarytool submit "$notary_archive" \
  --key "$notary_key" \
  --key-id "$APPLE_NOTARIZE_KEY_ID" \
  --issuer "$APPLE_NOTARIZE_ISSUER_ID" \
  --wait

# A bare executable cannot carry a stapled notarization ticket and is not an
# app/package type that spctl can assess. The strict codesign verification above
# and notarytool's blocking Accepted result are the applicable release checks.
