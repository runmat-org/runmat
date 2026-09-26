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

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

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
        serializer.serialize_str(&self.to_string())
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
mod tests {
    use super::Digest;

    #[test]
    fn segmented_hash_is_boundary_independent_and_order_sensitive() {
        let bytes = (0_u8..=255).collect::<Vec<_>>();
        for width in [1, 3, 64, 255, 256, 257] {
            assert_eq!(
                Digest::sha256_parts(bytes.chunks(width)),
                Digest::sha256(&bytes)
            );
        }
        assert_eq!(
            Digest::sha256_parts([b"".as_slice(), b"abc", b""]),
            Digest::sha256(b"abc")
        );
        assert_eq!(
            Digest::sha256_parts(std::iter::empty::<&[u8]>()),
            Digest::sha256(b"")
        );
        assert_ne!(
            Digest::sha256_parts([b"abc", b"def"]),
            Digest::sha256_parts([b"def", b"abc"])
        );
    }
}
