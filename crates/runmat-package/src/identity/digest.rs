use crate::IdentityError;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DigestAlgorithm {
    Sha256,
}

impl DigestAlgorithm {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentDigest {
    algorithm: DigestAlgorithm,
    bytes: [u8; 32],
}

impl ContentDigest {
    pub fn sha256(bytes: impl AsRef<[u8]>) -> Self {
        let digest = Sha256::digest(bytes.as_ref());
        let mut result = [0_u8; 32];
        result.copy_from_slice(&digest);
        Self {
            algorithm: DigestAlgorithm::Sha256,
            bytes: result,
        }
    }

    pub const fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

impl Display for ContentDigest {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:", self.algorithm.label())?;
        for byte in self.bytes {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for ContentDigest {
    type Err = IdentityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let Some((algorithm, encoded)) = value.split_once(':') else {
            return Err(invalid_digest(value, "missing algorithm label"));
        };
        if algorithm != DigestAlgorithm::Sha256.label() {
            return Err(invalid_digest(value, "unsupported digest algorithm"));
        }
        if encoded.len() != 64 {
            return Err(invalid_digest(
                value,
                "SHA-256 must contain exactly 64 lowercase hexadecimal digits",
            ));
        }
        if encoded
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        {
            return Err(invalid_digest(
                value,
                "digest bytes must use lowercase hexadecimal",
            ));
        }
        let mut bytes = [0_u8; 32];
        for (index, slot) in bytes.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
                .map_err(|_| invalid_digest(value, "invalid hexadecimal digest"))?;
        }
        Ok(Self {
            algorithm: DigestAlgorithm::Sha256,
            bytes,
        })
    }
}

impl Serialize for ContentDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ContentDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

fn invalid_digest(value: &str, reason: &'static str) -> IdentityError {
    IdentityError::InvalidDigest {
        value: value.to_string(),
        reason,
    }
}
