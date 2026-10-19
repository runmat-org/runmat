//! Portable paths within an explicitly bound source or workspace root.
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    path::Path,
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(try_from = "String", into = "String")]
pub struct NormalizedRelativePath(String);

impl NormalizedRelativePath {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, RelativePathError> {
        let path = path.as_ref();
        let Some(raw) = path.to_str() else {
            return Err(invalid_path(&path.to_string_lossy(), "must be valid UTF-8"));
        };
        let portable = raw.replace('\\', "/");
        if portable.is_empty() {
            return Ok(Self(".".to_string()));
        }
        if portable.starts_with('/') || portable.as_bytes().get(1).is_some_and(|byte| *byte == b':')
        {
            return Err(invalid_path(&portable, "must be relative"));
        }
        let mut segments = Vec::new();
        for segment in portable.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    return Err(invalid_path(&portable, "must not contain parent traversal"));
                }
                _ if segment
                    .chars()
                    .any(|character| character.is_control() || character == ':') =>
                {
                    return Err(invalid_path(&portable, "contains a non-portable character"));
                }
                _ => segments.push(segment),
            }
        }
        Ok(Self(if segments.is_empty() {
            ".".to_string()
        } else {
            segments.join("/")
        }))
    }

    pub fn is_root(&self) -> bool {
        self.0 == "."
    }

    pub fn file(path: impl AsRef<Path>) -> Result<Self, RelativePathError> {
        let path = Self::new(path)?;
        if path.is_root() {
            return Err(invalid_path(path.as_str(), "must name a file"));
        }
        Ok(path)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for NormalizedRelativePath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for NormalizedRelativePath {
    type Err = RelativePathError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(Path::new(value))
    }
}

impl TryFrom<String> for NormalizedRelativePath {
    type Error = RelativePathError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<NormalizedRelativePath> for String {
    fn from(value: NormalizedRelativePath) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid relative path `{value}`: {reason}")]
pub struct RelativePathError {
    pub value: String,
    pub reason: &'static str,
}
fn invalid_path(value: &str, reason: &'static str) -> RelativePathError {
    RelativePathError {
        value: value.to_owned(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_paths_normalize_portably_without_a_synthetic_root() {
        assert_eq!(
            NormalizedRelativePath::file("./studies//main.fea")
                .unwrap()
                .as_str(),
            "studies/main.fea"
        );
        assert_eq!(
            NormalizedRelativePath::file("project/main.fea")
                .unwrap()
                .as_str(),
            "project/main.fea"
        );
        assert!(NormalizedRelativePath::new(".").unwrap().is_root());
        for bad in [
            "/main.fea",
            "../outside.fea",
            "studies/../main.fea",
            "C:\\main.fea",
            "bad\nname.fea",
        ] {
            assert!(NormalizedRelativePath::file(bad).is_err(), "{bad}");
        }
        assert!(NormalizedRelativePath::file(".").is_err());
    }
}
