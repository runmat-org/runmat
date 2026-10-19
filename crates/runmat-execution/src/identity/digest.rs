use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::ContractError;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Digest([u8; 32]);

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

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
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
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
                .map_err(|_| ContractError::invalid("digest", "invalid hexadecimal"))?;
        }
        Ok(Self(bytes))
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
