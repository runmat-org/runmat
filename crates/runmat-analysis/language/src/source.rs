use crate::SourceDiagnostic;
use runmat_types::{CapabilitySet, EffectSet, OperationParameter, OperationTypePattern};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

macro_rules! source_identity {
    ($name:ident, $error:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, SourceIdentityError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(SourceIdentityError($error));
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

source_identity!(LogicalSourcePath, "logical source path");
source_identity!(SourceSymbol, "source symbol");
source_identity!(ExecutionProfileIdentity, "execution profile");

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0} must not be empty")]
pub struct SourceIdentityError(&'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceDigest(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModuleRequest {
    pub referrer: Option<LogicalSourcePath>,
    pub logical_path: LogicalSourcePath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceModule {
    pub logical_path: LogicalSourcePath,
    pub source_digest: SourceDigest,
    pub source: Vec<u8>,
}

pub trait FeaModuleSource {
    fn load(&self, request: &ModuleRequest) -> Result<SourceModule, SourceDiagnostic>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionRequest {
    pub module_path: LogicalSourcePath,
    pub symbol: SourceSymbol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionFacts {
    pub module_path: LogicalSourcePath,
    pub symbol: SourceSymbol,
    pub source_digest: SourceDigest,
    pub inputs: Vec<OperationParameter>,
    pub outputs: Vec<OperationTypePattern>,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
}

pub trait RunMatFunctionSource {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryInspectionRequest {
    pub referrer: LogicalSourcePath,
    pub logical_path: LogicalSourcePath,
    /// Authored source units used to normalize exact topology identity.
    pub source_units: runmat_geometry_core::UnitSystem,
    pub expected_content_digest: Option<SourceDigest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GeometryInspectionCapability {
    Topology,
    PersistentIds,
    Parameters,
    EngineeringProperties,
    MaterialCandidates,
    Regeneration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryInspection {
    pub content_digest: SourceDigest,
    pub topology_digest: SourceDigest,
    pub capabilities: Vec<GeometryInspectionCapability>,
    pub exchange: Option<runmat_geometry_core::Ap242ExchangeInventory>,
}

pub trait GeometryInspectionSource {
    fn inspect(
        &self,
        request: &GeometryInspectionRequest,
    ) -> Result<GeometryInspection, SourceDiagnostic>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfileRequest {
    pub profile: ExecutionProfileIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionPlacement {
    Inline,
    Isolated,
    Durable,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfileFacts {
    pub profile: ExecutionProfileIdentity,
    pub placement: ExecutionPlacement,
    pub capabilities: CapabilitySet,
}

pub trait ExecutionProfileSource {
    fn resolve(
        &self,
        request: &ExecutionProfileRequest,
    ) -> Result<ExecutionProfileFacts, SourceDiagnostic>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_identities_reject_empty_decoded_values() {
        assert!(LogicalSourcePath::new(" ").is_err());
        assert!(serde_json::from_str::<SourceSymbol>("\"\"").is_err());
        assert!(serde_json::from_str::<ExecutionProfileIdentity>("\"  \"").is_err());
    }
}
