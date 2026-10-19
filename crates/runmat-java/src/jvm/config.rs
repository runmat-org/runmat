use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::JvmError;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JvmVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub raw: String,
}

impl JvmVersion {
    pub fn parse(raw: impl Into<String>) -> Result<Self, JvmError> {
        let raw = raw.into();
        let unquoted = raw.trim().trim_matches('"');
        let numeric = unquoted.strip_prefix("1.").unwrap_or(unquoted);
        let mut components = numeric.split(|character: char| !character.is_ascii_digit());
        let first = components
            .next()
            .filter(|component| !component.is_empty())
            .ok_or_else(|| {
                JvmError::InvalidConfiguration(format!("invalid Java version `{raw}`"))
            })?;
        let parsed_first = first
            .parse::<u16>()
            .map_err(|_| JvmError::InvalidConfiguration(format!("invalid Java version `{raw}`")))?;
        let major = parsed_first;
        let minor = components
            .next()
            .filter(|component| !component.is_empty())
            .and_then(|component| component.parse().ok())
            .unwrap_or(0);
        let patch = components
            .next()
            .filter(|component| !component.is_empty())
            .and_then(|component| component.parse().ok())
            .unwrap_or(0);
        Ok(Self {
            major,
            minor,
            patch,
            raw,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JvmConfig {
    pub minimum_major: u16,
    pub maximum_major: Option<u16>,
    pub bootstrap_classpath: Vec<PathBuf>,
    pub options: Vec<String>,
}

impl Default for JvmConfig {
    fn default() -> Self {
        Self {
            minimum_major: 8,
            maximum_major: None,
            bootstrap_classpath: Vec::new(),
            options: Vec::new(),
        }
    }
}

impl JvmConfig {
    pub fn validate(&self) -> Result<(), JvmError> {
        if self.minimum_major == 0 {
            return Err(JvmError::InvalidConfiguration(
                "minimum Java version must be non-zero".into(),
            ));
        }
        if self
            .maximum_major
            .is_some_and(|maximum| maximum < self.minimum_major)
        {
            return Err(JvmError::InvalidConfiguration(
                "maximum Java version is older than the minimum".into(),
            ));
        }
        if self.options.iter().any(|option| option.contains('\0')) {
            return Err(JvmError::InvalidConfiguration(
                "JVM options cannot contain a null character".into(),
            ));
        }
        Ok(())
    }

    pub fn accepts(&self, version: &JvmVersion) -> bool {
        version.major >= self.minimum_major
            && self
                .maximum_major
                .is_none_or(|maximum| version.major <= maximum)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn launch_options(&self) -> Vec<String> {
        let mut options = self.options.clone();
        if !self.bootstrap_classpath.is_empty() {
            let classpath = std::env::join_paths(&self.bootstrap_classpath)
                .map(|paths| paths.to_string_lossy().into_owned())
                .unwrap_or_default();
            options.push(format!("-Djava.class.path={classpath}"));
        }
        options
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legacy_and_modern_versions() {
        assert_eq!(JvmVersion::parse("1.8.0_402").unwrap().major, 8);
        let modern = JvmVersion::parse("21.0.4+7").unwrap();
        assert_eq!((modern.major, modern.minor, modern.patch), (21, 0, 4));
    }

    #[test]
    fn validates_version_range() {
        let config = JvmConfig {
            minimum_major: 17,
            maximum_major: Some(21),
            ..JvmConfig::default()
        };
        assert!(config.accepts(&JvmVersion::parse("21.0.1").unwrap()));
        assert!(!config.accepts(&JvmVersion::parse("22").unwrap()));
    }
}
