//! Portable declarations for paired native application and test products.
//!
//! These codecs establish structural consistency only. The linked receiver
//! must compare embedded headers to its actual compiled entry map and issue
//! results from execution; decoded bytes never confer executable authority.

mod binding;
mod protocol;

pub use binding::{
    NativeCompiledEntry, NativePairedArtifactManifest, NativePairedBuildBinding,
    NativePairedProductHeader, NativePairedProductRole, NativeTestEntry,
    NATIVE_PAIRED_TEST_SCHEMA_VERSION,
};
pub use protocol::{
    NativeTargetTestFailure, NativeTargetTestRequest, NativeTargetTestResult,
    NativeTargetTestStatus, NativeTargetTestUsage, NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
};

use serde::{de::DeserializeOwned, Serialize};
use std::io::{self, Write};

use crate::{ArtifactError, ArtifactResult};

/// Maximum canonical header, request or result payload, excluding framing.
pub const MAX_NATIVE_TARGET_TEST_WIRE_BYTES: usize = 8 * 1024 * 1024;
const MAX_WIRE_BYTES: usize = MAX_NATIVE_TARGET_TEST_WIRE_BYTES;

fn canonical_bytes<T: Serialize>(value: &T) -> ArtifactResult<Vec<u8>> {
    let mut writer = BoundedWireWriter::default();
    if let Err(error) = serde_json::to_writer(&mut writer, value) {
        return if writer.exceeded {
            Err(ArtifactError::Limit("native target-test wire bytes".into()))
        } else {
            Err(ArtifactError::Encoding(error.to_string()))
        };
    }
    Ok(writer.bytes)
}

#[derive(Default)]
struct BoundedWireWriter {
    bytes: Vec<u8>,
    exceeded: bool,
}

impl Write for BoundedWireWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_WIRE_BYTES.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("native target-test wire byte limit"));
        }
        let required = self.bytes.len() + bytes.len();
        if required > self.bytes.capacity() {
            let target = required
                .max(self.bytes.capacity().saturating_mul(2))
                .min(MAX_WIRE_BYTES);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(io::Error::other)?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn decode_canonical<T: DeserializeOwned + Serialize>(bytes: &[u8]) -> ArtifactResult<T> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(ArtifactError::Limit("native target-test wire bytes".into()));
    }
    let value =
        serde_json::from_slice(bytes).map_err(|e| ArtifactError::Encoding(e.to_string()))?;
    if canonical_bytes(&value)? != bytes {
        return Err(ArtifactError::Invalid(
            "noncanonical native target-test wire bytes".into(),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::{canonical_bytes, decode_canonical, MAX_WIRE_BYTES};
    use crate::ArtifactError;

    #[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        suite: String,
    }

    #[test]
    fn canonical_codec_roundtrip_and_rejections() {
        let fixture = Fixture {
            suite: "case".into(),
        };
        let canonical = canonical_bytes(&fixture).expect("encode");
        assert_eq!(canonical, br#"{"suite":"case"}"#);
        assert_eq!(
            decode_canonical::<Fixture>(&canonical).expect("decode"),
            fixture
        );
        assert!(matches!(
            decode_canonical::<Fixture>(br#"{ "suite":"case"}"#),
            Err(ArtifactError::Invalid(_))
        ));
        assert!(matches!(
            decode_canonical::<Fixture>(br#"{"suite":"case","extra":1}"#),
            Err(ArtifactError::Encoding(_))
        ));
        let oversized = Fixture {
            suite: "x".repeat(MAX_WIRE_BYTES),
        };
        assert!(matches!(
            canonical_bytes(&oversized),
            Err(ArtifactError::Limit(_))
        ));
        assert!(matches!(
            decode_canonical::<Fixture>(&vec![b' '; MAX_WIRE_BYTES + 1]),
            Err(ArtifactError::Limit(_))
        ));
    }
}
