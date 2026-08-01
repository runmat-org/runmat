use crate::ManifestError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HostCapability {
    BrowserFilesystem,
    Network,
    WebGpu,
    Worker,
    SharedMemory,
    NativeLibrary,
    Mex,
    Jvm,
    Subprocess,
}

impl HostCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BrowserFilesystem => "browser-filesystem",
            Self::Network => "network",
            Self::WebGpu => "webgpu",
            Self::Worker => "worker",
            Self::SharedMemory => "shared-memory",
            Self::NativeLibrary => "native-library",
            Self::Mex => "mex",
            Self::Jvm => "jvm",
            Self::Subprocess => "subprocess",
        }
    }
}

impl Display for HostCapability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for HostCapability {
    type Err = ManifestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "browser-filesystem" => Ok(Self::BrowserFilesystem),
            "network" => Ok(Self::Network),
            "webgpu" => Ok(Self::WebGpu),
            "worker" => Ok(Self::Worker),
            "shared-memory" => Ok(Self::SharedMemory),
            "native-library" => Ok(Self::NativeLibrary),
            "mex" => Ok(Self::Mex),
            "jvm" => Ok(Self::Jvm),
            "subprocess" => Ok(Self::Subprocess),
            _ => Err(ManifestError::InvalidCapability(value.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "kebab-case")]
pub enum TargetPredicate {
    Triple(String),
    Capability(HostCapability),
}

impl Display for TargetPredicate {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Triple(triple) => formatter.write_str(triple),
            Self::Capability(capability) => write!(formatter, "capability:{capability}"),
        }
    }
}

impl FromStr for TargetPredicate {
    type Err = ManifestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if let Some(capability) = value.strip_prefix("capability:") {
            return Ok(Self::Capability(capability.parse()?));
        }
        if value.is_empty()
            || value.chars().any(char::is_whitespace)
            || value.contains(['/', '\\', ':'])
        {
            return Err(ManifestError::InvalidTarget {
                value: value.to_string(),
                reason: "expected a normalized target triple or `capability:<name>`".to_string(),
            });
        }
        let segment_count = value.split('-').count();
        if segment_count < 3 {
            return Err(ManifestError::InvalidTarget {
                value: value.to_string(),
                reason: "target triples must contain at least architecture, vendor, and system"
                    .to_string(),
            });
        }
        Ok(Self::Triple(value.to_ascii_lowercase()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetEnvironment {
    pub triple: String,
    pub capabilities: BTreeSet<HostCapability>,
}

impl TargetEnvironment {
    pub fn supports(&self, predicate: &TargetPredicate) -> bool {
        match predicate {
            TargetPredicate::Triple(triple) => &self.triple == triple,
            TargetPredicate::Capability(capability) => self.capabilities.contains(capability),
        }
    }
}
