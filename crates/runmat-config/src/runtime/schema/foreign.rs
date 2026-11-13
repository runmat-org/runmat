use std::collections::BTreeSet;
use std::num::{NonZeroU16, NonZeroU64};
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ForeignConfig {
    #[serde(default)]
    pub mex: MexConfig,
    #[serde(default)]
    pub native: NativeFfiConfig,
    #[serde(default)]
    pub java: JavaConfig,
    #[serde(default)]
    pub python: PythonConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonExecutionModeConfig {
    #[default]
    InProcess,
    OutOfProcess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonConfig {
    #[serde(default)]
    pub executable: Option<PathBuf>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default = "default_python_minimum_version")]
    pub minimum_version: String,
    #[serde(default)]
    pub maximum_version: Option<String>,
    #[serde(default)]
    pub execution_mode: PythonExecutionModeConfig,
    #[serde(default)]
    pub exact_version: Option<PythonExactVersion>,
    #[serde(default)]
    pub required_abi_tag: Option<String>,
    #[serde(default)]
    pub required_platform_tag: Option<String>,
    #[serde(default)]
    pub module_paths: Vec<PathBuf>,
    /// Present only when the capture owner proved these paths came from its
    /// installed Python bundle, rather than an explicit user module path.
    #[serde(default)]
    pub installed_module_paths: Option<PythonInstalledModulePaths>,
    #[serde(default)]
    pub artifact_identities: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonExactVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonInstalledModulePaths {
    pub artifact_identities: BTreeSet<String>,
    /// Positions in PythonConfig.module_paths certified by the installed
    /// bundle owner. Paths outside these positions remain explicit policy.
    /// Legacy certificates without this field are rejected at typed decode;
    /// they cannot authorize selected-worker path rebasing.
    pub installed_path_indices: Vec<u64>,
}

impl Default for PythonConfig {
    fn default() -> Self {
        Self {
            executable: None,
            version: None,
            minimum_version: default_python_minimum_version(),
            maximum_version: None,
            execution_mode: PythonExecutionModeConfig::InProcess,
            exact_version: None,
            required_abi_tag: None,
            required_platform_tag: None,
            module_paths: Vec::new(),
            installed_module_paths: None,
            artifact_identities: BTreeSet::new(),
        }
    }
}

fn default_python_minimum_version() -> String {
    "3.9".into()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JavaConfig {
    #[serde(default)]
    pub home: Option<PathBuf>,
    #[serde(default = "default_java_minimum_version")]
    pub minimum_version: NonZeroU16,
    #[serde(default)]
    pub maximum_version: Option<NonZeroU16>,
    #[serde(default)]
    pub classpath: Vec<PathBuf>,
    #[serde(default)]
    pub options: Vec<String>,
}

impl Default for JavaConfig {
    fn default() -> Self {
        Self {
            home: None,
            minimum_version: default_java_minimum_version(),
            maximum_version: None,
            classpath: Vec::new(),
            options: Vec::new(),
        }
    }
}

fn default_java_minimum_version() -> NonZeroU16 {
    NonZeroU16::new(8).expect("Java minimum version is nonzero")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeFfiConfig {
    #[serde(default)]
    pub isolation: NativeFfiIsolation,
    #[serde(default)]
    pub timeout_ms: Option<NonZeroU64>,
    /// Exact native deadline. New captures use this field; timeout_ms remains
    /// the legacy input representation. Both may not be specified together.
    #[serde(default)]
    pub timeout: Option<NativeFfiTimeout>,
}

impl<'de> Deserialize<'de> for NativeFfiConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            #[serde(default)]
            isolation: NativeFfiIsolation,
            #[serde(default)]
            timeout_ms: Option<NonZeroU64>,
            #[serde(default)]
            timeout: Option<NativeFfiTimeout>,
        }
        let wire = Wire::deserialize(deserializer)?;
        if wire.timeout_ms.is_some() && wire.timeout.is_some() {
            return Err(serde::de::Error::custom(
                "native timeout and timeout_ms are mutually exclusive",
            ));
        }
        Ok(Self {
            isolation: wire.isolation,
            timeout_ms: wire.timeout_ms,
            timeout: wire.timeout,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct NativeFfiTimeout {
    secs: u64,
    nanos: u32,
}

impl NativeFfiTimeout {
    pub fn new(duration: Duration) -> Option<Self> {
        (!duration.is_zero()).then_some(Self {
            secs: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        })
    }

    pub fn duration(self) -> Duration {
        Duration::new(self.secs, self.nanos)
    }
}

impl<'de> Deserialize<'de> for NativeFfiTimeout {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            secs: u64,
            nanos: u32,
        }
        let wire = Wire::deserialize(deserializer)?;
        if wire.nanos >= 1_000_000_000 || (wire.secs == 0 && wire.nanos == 0) {
            return Err(serde::de::Error::custom(
                "native timeout must be nonzero with nanos below one second",
            ));
        }
        Ok(Self {
            secs: wire.secs,
            nanos: wire.nanos,
        })
    }
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
            timeout: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        assert!(toml::from_str::<ForeignConfig>("[native.timeout]\nsecs = 0\nnanos = 0").is_err());
        assert!(
            toml::from_str::<ForeignConfig>("[native.timeout]\nsecs = 0\nnanos = 1000000000")
                .is_err()
        );
        assert!(toml::from_str::<ForeignConfig>(
            "[native]\ntimeout_ms = 1\n[native.timeout]\nsecs = 0\nnanos = 1"
        )
        .is_err());
    }

    #[test]
    fn exact_native_deadline_and_python_pins_roundtrip() {
        let policy: ForeignConfig = toml::from_str("[native.timeout]\nsecs = 17\nnanos = 123456789\n[python]\nversion = '3.13'\nexact_version = { major = 3, minor = 13, patch = 2 }\nrequired_abi_tag = 'cp313'\nrequired_platform_tag = 'linux_x86_64'\nmodule_paths = ['/private/wheel']\nartifact_identities = ['wheel:abc']").unwrap();
        let encoded = toml::to_string(&policy).unwrap();
        let decoded: ForeignConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.native.timeout.unwrap().duration(),
            Duration::new(17, 123456789)
        );
        assert_eq!(decoded.python.exact_version.unwrap().patch, 2);
        assert_eq!(decoded.python.required_abi_tag.as_deref(), Some("cp313"));
        assert_eq!(
            decoded.python.required_platform_tag.as_deref(),
            Some("linux_x86_64")
        );
        assert_eq!(
            decoded.python.module_paths,
            vec![PathBuf::from("/private/wheel")]
        );
        assert!(decoded.python.artifact_identities.contains("wheel:abc"));
    }

    #[test]
    fn native_libraries_default_to_process_isolation() {
        let config = ForeignConfig::default();
        assert_eq!(config.native.isolation, NativeFfiIsolation::Process);
        assert_eq!(config.native.timeout_ms, None);
    }

    #[test]
    fn java_configuration_is_explicit_and_version_bounded() {
        let config: ForeignConfig = toml::from_str(
            "[java]\nminimum_version = 17\nmaximum_version = 21\nclasspath = ['lib/runtime.jar']\noptions = ['-Xmx512m']",
        )
        .unwrap();
        assert_eq!(config.java.minimum_version.get(), 17);
        assert_eq!(config.java.maximum_version.unwrap().get(), 21);
        assert_eq!(
            config.java.classpath,
            vec![PathBuf::from("lib/runtime.jar")]
        );
        assert_eq!(config.java.options, vec!["-Xmx512m"]);
        assert!(toml::from_str::<ForeignConfig>("[java]\nminimum_version = 0").is_err());
    }

    #[test]
    fn python_configuration_keeps_version_and_execution_policy_explicit() {
        let config: ForeignConfig = toml::from_str(
            "[python]\nversion = '3.13'\nminimum_version = '3.9'\nmaximum_version = '3.13'\nexecution_mode = 'out_of_process'",
        )
        .unwrap();
        assert_eq!(config.python.version.as_deref(), Some("3.13"));
        assert_eq!(config.python.minimum_version, "3.9");
        assert_eq!(config.python.maximum_version.as_deref(), Some("3.13"));
        assert_eq!(
            config.python.execution_mode,
            PythonExecutionModeConfig::OutOfProcess
        );
        assert!(config.python.installed_module_paths.is_none());
        let legacy = serde_json::json!({"artifact_identities": ["wheel-id"]});
        assert!(serde_json::from_value::<PythonInstalledModulePaths>(legacy)
            .unwrap_err()
            .to_string()
            .contains("installed_path_indices"));
    }
}
