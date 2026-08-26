use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ForeignConfig {
    #[serde(default)]
    pub mex: MexConfig,
    #[serde(default)]
    pub native: NativeFfiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFfiConfig {
    #[serde(default)]
    pub isolation: NativeFfiIsolation,
    #[serde(default)]
    pub timeout_ms: Option<NonZeroU64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NativeFfiIsolation {
    #[default]
    Process,
    InProcess,
}

impl Default for NativeFfiConfig {
    fn default() -> Self {
        Self {
            isolation: NativeFfiIsolation::Process,
            timeout_ms: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MexConfig {
    #[serde(default)]
    pub unmanifested: UnmanifestedMexPolicy,
    #[serde(default)]
    pub timeout_ms: Option<NonZeroU64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UnmanifestedMexPolicy {
    #[default]
    Isolate,
    Deny,
}

impl Default for MexConfig {
    fn default() -> Self {
        Self {
            unmanifested: UnmanifestedMexPolicy::Isolate,
            timeout_ms: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_defaults_to_isolation_without_an_implicit_deadline() {
        let config = MexConfig::default();
        assert_eq!(config.unmanifested, UnmanifestedMexPolicy::Isolate);
        assert_eq!(config.timeout_ms, None);
    }

    #[test]
    fn zero_timeout_is_rejected_during_decode() {
        assert!(toml::from_str::<ForeignConfig>("[mex]\ntimeout_ms = 0").is_err());
        assert!(toml::from_str::<ForeignConfig>("[native]\ntimeout_ms = 0").is_err());
    }

    #[test]
    fn native_libraries_default_to_process_isolation() {
        let config = ForeignConfig::default();
        assert_eq!(config.native.isolation, NativeFfiIsolation::Process);
        assert_eq!(config.native.timeout_ms, None);
    }
}
