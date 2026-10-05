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
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = [0_u8; 71];
        text[..7].copy_from_slice(b"sha256:");
        for (index, byte) in self.0.iter().copied().enumerate() {
            text[7 + index * 2] = HEX[usize::from(byte >> 4)];
            text[8 + index * 2] = HEX[usize::from(byte & 0x0f)];
        }
        serializer.serialize_str(std::str::from_utf8(&text).expect("ASCII digest encoding"))
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
}
