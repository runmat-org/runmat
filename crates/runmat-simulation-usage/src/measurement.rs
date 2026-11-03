use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies a published numerical counting schedule. Parsing validates its
/// wire identity; the numerical owner must separately implement that schedule.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct MeasurementVersion(String);

impl MeasurementVersion {
    pub fn parse(value: &str) -> Result<Self, InvalidMeasurementVersion> {
        let bytes = value.as_bytes();
        if bytes.len() > 64
            || !bytes
                .first()
                .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            || !bytes.iter().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'_' | b'-')
            })
        {
            return Err(InvalidMeasurementVersion);
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for MeasurementVersion {
    type Error = InvalidMeasurementVersion;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}
impl From<MeasurementVersion> for String {
    fn from(value: MeasurementVersion) -> Self {
        value.0
    }
}
impl fmt::Display for MeasurementVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidMeasurementVersion;
impl fmt::Display for InvalidMeasurementVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("measurement version requires 1–64 lowercase ASCII letters, digits, '.', '_' or '-', starting with a letter or digit")
    }
}
impl std::error::Error for InvalidMeasurementVersion {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_identity_is_bounded_and_round_trips() {
        for value in ["rsua-1", "1", "a.b_c-2"] {
            let version = MeasurementVersion::parse(value).unwrap();
            assert_eq!(
                serde_json::from_str::<MeasurementVersion>(
                    &serde_json::to_string(&version).unwrap()
                )
                .unwrap(),
                version
            );
        }
        for value in [
            "",
            ".rsua-1",
            "RSUA-1",
            "rsua/1",
            "rsua 1",
            "é",
            &"a".repeat(65),
        ] {
            assert!(MeasurementVersion::parse(value).is_err());
        }
    }
}
