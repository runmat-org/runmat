use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonExecutionMode {
    InProcess,
    OutOfProcess,
}

impl PythonExecutionMode {
    pub const fn as_compatibility_name(self) -> &'static str {
        match self {
            Self::InProcess => "InProcess",
            Self::OutOfProcess => "OutOfProcess",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonStatus {
    NotLoaded,
    Loaded,
    Terminated,
}

impl PythonStatus {
    pub const fn as_compatibility_name(self) -> &'static str {
        match self {
            Self::NotLoaded => "NotLoaded",
            Self::Loaded => "Loaded",
            Self::Terminated => "Terminated",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PythonVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl std::fmt::Display for PythonVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonDiscoveryRequest {
    pub executable: Option<PathBuf>,
    pub version: Option<(u16, u16)>,
    pub minimum_version: (u16, u16),
    pub maximum_version: Option<(u16, u16)>,
}

impl Default for PythonDiscoveryRequest {
    fn default() -> Self {
        Self {
            executable: None,
            version: None,
            minimum_version: (3, 9),
            maximum_version: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonInstallation {
    pub version: PythonVersion,
    pub executable: PathBuf,
    pub library: PathBuf,
    pub home: PathBuf,
    pub prefix: PathBuf,
}
