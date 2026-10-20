use std::fmt::{Display, Formatter};
use std::io::{self, Write};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest([u8; 32]);

/// Incremental identity writer for callers whose legacy byte encoding is a
/// sequence of fixed fields and serialized fields. It retains no byte buffer.
pub struct DigestStream(Sha256);

impl DigestStream {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    pub fn finish(self) -> Digest {
        Digest(self.0.finalize().into())
    }
}

impl Default for DigestStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Write for DigestStream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl minicbor::encode::Write for DigestStream {
    type Error = std::convert::Infallible;

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.update(bytes);
        Ok(())
    }
}

impl Digest {
    pub fn sha256(bytes: impl AsRef<[u8]>) -> Self {
        Self(Sha256::digest(bytes.as_ref()).into())
    }

    /// Hash ordered byte segments as one byte stream, without concatenating them.
    /// Segment boundaries are not part of the identity.
    pub fn sha256_parts(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> Self {
        let mut hash = Sha256::new();
        for part in parts {
            hash.update(part.as_ref());
        }
        Self(hash.finalize().into())
    }

    /// Hash the same ordered byte stream while probing between bounded blocks.
    /// The caller owns cancellation and resource errors; no partial digest is returned.
    pub fn sha256_parts_with_checkpoint<E>(
        parts: impl IntoIterator<Item = impl AsRef<[u8]>>,
        mut checkpoint: impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut hash = Sha256::new();
        for part in parts {
            checkpoint()?;
            for block in part.as_ref().chunks(1024 * 1024) {
                checkpoint()?;
                hash.update(block);
            }
        }
        checkpoint()?;
        Ok(Self(hash.finalize().into()))
    }

    /// Hash the existing JSON identity encoding without retaining a second
    /// serialized buffer. This does not certify allocations in the caller or
    /// in a placement service consuming the resulting digest.
    pub fn sha256_serialized<T: Serialize>(value: &T) -> serde_json::Result<Self> {
        let mut writer = DigestStream::new();
        serde_json::to_writer(&mut writer, value)?;
        Ok(writer.finish())
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Recognize the exact canonical presentation without allocating an
    /// error. Non-digest candidate names remain text identities.
    pub fn parse_canonical(value: &str) -> Option<Self> {
        let encoded = value.strip_prefix("sha256:")?.as_bytes();
        if encoded.len() != 64 {
            return None;
        }
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            let high = lowercase_hex(encoded[index * 2])?;
            let low = lowercase_hex(encoded[index * 2 + 1])?;
            *byte = (high << 4) | low;
        }
        Some(Self(bytes))
    }
}

fn lowercase_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests;

impl Display for Digest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for Digest {
    type Err = ContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let encoded = value
            .strip_prefix("sha256:")
            .ok_or_else(|| ContractError::invalid("digest", "missing `sha256:` prefix"))?;
        if encoded.len() != 64
            || encoded
                .bytes()
                .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err(ContractError::invalid(
                "digest",
                "expected 64 lowercase hexadecimal digits",
            ));
        }
        Self::parse_canonical(value)
            .ok_or_else(|| ContractError::invalid("digest", "invalid hexadecimal"))
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        super::hex::serialize::<71, S>("sha256:", &self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod wire_tests {
    use super::Digest;
    #[test]
    fn fixed_wire_encoding_preserves_every_byte_and_original_lowercase_spelling() {
        for start in 0u8..=255 {
            let bytes = std::array::from_fn(|i| start.wrapping_add(i as u8));
            let digest = Digest::from_bytes(bytes);
            let expected = format!("\"{}\"", digest);
            let wire = serde_json::to_string(&digest).unwrap();
            assert_eq!(wire, expected);
            assert_eq!(serde_json::from_str::<Digest>(&wire).unwrap(), digest);
        }
    }
}

#[cfg(test)]
mod streaming_serialization_tests {
    use super::*;

    #[test]
    fn streaming_json_keeps_existing_digest_schema() {
        let value = ("host", Digest::sha256(b"source"), 17_u64);
        let legacy = Digest::sha256(serde_json::to_vec(&value).unwrap());
        assert_eq!(Digest::sha256_serialized(&value).unwrap(), legacy);
    }

    #[test]
    fn serialized_digest_retains_exact_lowercase_presentation() {
        let value = Digest::sha256(b"no allocation in digest presentation");
        assert_eq!(
            serde_json::to_string(&value).unwrap(),
            serde_json::to_string(&value.to_string()).unwrap()
        );
    }

    #[test]
    fn generic_streaming_serialization_preserves_failure() {
        use serde::ser::Error as _;

        struct Fails;
        impl Serialize for Fails {
            fn serialize<S>(&self, _: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(S::Error::custom("deliberate serialization failure"))
            }
        }
        assert!(Digest::sha256_serialized(&Fails).is_err());
    }

    #[test]
    fn optional_canonical_decoder_agrees_with_strict_parser() {
        let canonical = Digest::sha256(b"candidate").to_string();
        for value in [
            canonical.as_str(),
            "candidate",
            "sha256:ABCDEF",
            "sha256:é",
            "sha256:000000000000000000000000000000000000000000000000000000000000000g",
        ] {
            assert_eq!(Digest::parse_canonical(value), value.parse::<Digest>().ok());
        }
    }
}
