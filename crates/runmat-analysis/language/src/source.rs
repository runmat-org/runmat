use crate::SourceDiagnostic;
use runmat_types::{
    AnalysisValueType, CapabilitySet, EffectSet, OperationParameter, OperationTypePattern, Span,
    ValueFact, ValueKindFact,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

macro_rules! source_identity {
    ($name:ident, $error:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        #[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, SourceIdentityError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(SourceIdentityError::Empty($error));
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

/// A portable source path relative to the host-bound source root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct LogicalSourcePath(runmat_types::NormalizedRelativePath);
impl LogicalSourcePath {
    pub fn new(value: impl Into<String>) -> Result<Self, SourceIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SourceIdentityError::Empty("logical source path"));
        }
        runmat_types::NormalizedRelativePath::new(value)
            .map(Self)
            .map_err(Into::into)
    }
    pub fn file(value: impl Into<String>) -> Result<Self, SourceIdentityError> {
        let value = value.into();
        runmat_types::NormalizedRelativePath::file(value)
            .map(Self)
            .map_err(Into::into)
    }
    pub fn root() -> Self {
        Self(runmat_types::NormalizedRelativePath::new(".").expect("source root"))
    }
    pub fn is_root(&self) -> bool {
        self.0.is_root()
    }
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}
impl fmt::Display for LogicalSourcePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
impl<'de> Deserialize<'de> for LogicalSourcePath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
source_identity!(SourceSymbol, "source symbol");
source_identity!(ExecutionProfileIdentity, "execution profile");

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SourceIdentityError {
    #[error("{0} must not be empty")]
    Empty(&'static str),
    #[error(transparent)]
    InvalidPath(#[from] runmat_types::RelativePathError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
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
    pub declaration_span: Span,
    pub inputs: Vec<OperationParameter>,
    pub outputs: Vec<OperationTypePattern>,
    pub effects: EffectSet,
    #[serde(default)]
    pub cooperative_suspension: bool,
    pub capabilities: CapabilitySet,
}

pub trait RunMatFunctionSource {
    fn resolve(&self, request: &FunctionRequest) -> Result<FunctionFacts, SourceDiagnostic>;
}

pub fn operation_pattern_from_value_fact(fact: &ValueFact) -> OperationTypePattern {
    match (&fact.kind, &fact.shape) {
        (ValueKindFact::Numeric(_), _) if fact.is_scalar() => OperationTypePattern::NumericScalar,
        (ValueKindFact::Numeric(_), _) => OperationTypePattern::NumericArray,
        (ValueKindFact::Logical, _) if fact.is_scalar() => OperationTypePattern::LogicalScalar,
        (ValueKindFact::Logical, _) => OperationTypePattern::LogicalArray,
        (ValueKindFact::Struct(record), _) if fact.is_scalar() && record.fields_complete => {
            OperationTypePattern::Record(
                record
                    .fields
                    .iter()
                    .map(|(name, value)| (name.clone(), operation_pattern_from_value_fact(value)))
                    .collect(),
            )
        }
        (ValueKindFact::Character | ValueKindFact::String, _) if fact.is_scalar() => {
            OperationTypePattern::StringScalar
        }
        (ValueKindFact::Object(object), shape) if shape.element_count() == Some(1) => {
            match object
                .runtime_class
                .as_ref()
                .map(runmat_types::ClassIdentity::display_name)
            {
                Some("geometry.Generated") => {
                    OperationTypePattern::Analysis(AnalysisValueType::Geometry)
                }
                Some("geometry.Feature") => {
                    OperationTypePattern::Analysis(AnalysisValueType::GeometryFeature)
                }
                Some("geometry.RegionOutput") => {
                    OperationTypePattern::Analysis(AnalysisValueType::Region)
                }
                _ => OperationTypePattern::Any,
            }
        }
        _ => OperationTypePattern::Any,
    }
}

#[cfg(test)]
mod assessment_callable_fact_tests {
    use super::*;
    use runmat_types::StructFact;

    #[test]
    fn complete_scalar_record_retains_logical_result_fields_for_singletons() {
        let fields = std::collections::BTreeMap::from([
            (
                "evaluated".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            ),
            ("accepted".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ]);
        let fact = ValueFact::scalar(ValueKindFact::Struct(StructFact {
            fields,
            fields_complete: true,
            elements: Vec::new(),
            elements_complete: false,
        }));
        assert_eq!(
            operation_pattern_from_value_fact(&fact),
            OperationTypePattern::Record(std::collections::BTreeMap::from([
                ("evaluated".into(), OperationTypePattern::LogicalScalar),
                ("accepted".into(), OperationTypePattern::LogicalScalar),
            ]))
        );
        let mut explicit_singleton = fact.clone();
        explicit_singleton.shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
        let ValueKindFact::Struct(record) = &mut explicit_singleton.kind else {
            unreachable!()
        };
        for value in record.fields.values_mut() {
            value.shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
        }
        assert_eq!(
            operation_pattern_from_value_fact(&explicit_singleton),
            operation_pattern_from_value_fact(&fact)
        );
    }

    #[test]
    fn incomplete_record_is_not_a_proven_return_contract() {
        let fact = ValueFact::scalar(ValueKindFact::Struct(StructFact {
            fields: std::collections::BTreeMap::new(),
            fields_complete: false,
            elements: Vec::new(),
            elements_complete: false,
        }));
        assert_eq!(
            operation_pattern_from_value_fact(&fact),
            OperationTypePattern::Any
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryInspectionRequest {
    pub referrer: LogicalSourcePath,
    pub logical_path: LogicalSourcePath,
    /// Authored source units used to normalize exact topology identity.
    pub source_units: Option<runmat_geometry_core::LengthUnit>,
    pub expected_content_digest: Option<SourceDigest>,
    /// `interfaces = shared`: touching bodies join so coincident faces become shared interfaces.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub shared_interfaces: bool,
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
    /// Extents measured from the already imported preview, in canonical metres.
    /// Advisory planning evidence only; not an exact meshing bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_extent_m: Option<[f64; 3]>,
    pub content_digest: SourceDigest,
    pub topology_digest: SourceDigest,
    pub capabilities: Vec<GeometryInspectionCapability>,
    pub exchange: Option<runmat_geometry_core::Ap242ExchangeInventory>,
    /// Length unit the geometry was inspected in: the authored `units`, or the
    /// single length unit the file itself declares when `units` is omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_units: Option<runmat_geometry_core::LengthUnit>,
    #[serde(default)]
    pub unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence,
    /// Authored names in the exact model by region role (`vertex`, `edge`, `face`, `solid`), one
    /// entry per named entity (sorted), so a check can reject a `select name` that matches nothing
    /// or more entities than the region expects. Empty when the source carries no exact names.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub authored_names: std::collections::BTreeMap<String, Vec<String>>,
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

pub fn remote_execution_profile_facts(
    profiles: impl IntoIterator<Item = (String, bool)>,
) -> Result<Vec<ExecutionProfileFacts>, SourceIdentityError> {
    profiles
        .into_iter()
        .map(|(name, has_accelerator)| {
            let mut capabilities = std::collections::BTreeSet::from([
                runmat_types::CapabilityRequirement::DistributedRuntime,
                runmat_types::CapabilityRequirement::ParallelRuntime,
            ]);
            if has_accelerator {
                capabilities.insert(runmat_types::CapabilityRequirement::Accelerator);
            }
            Ok(ExecutionProfileFacts {
                profile: ExecutionProfileIdentity::new(name)?,
                placement: ExecutionPlacement::Remote,
                capabilities: CapabilitySet(capabilities),
            })
        })
        .collect()
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
