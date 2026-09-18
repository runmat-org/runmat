use std::collections::BTreeMap;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef, GeometryDigest,
    GeometryDocument, GeometryModel, GeometryObjectRef,
};
use runmat_geometry_io::ExactCadImportOptions;
use runmat_geometry_ops::{
    GeometryFeatureCapabilities, GeometryFeaturePlanLimits, GeometryFeatureResourceBounds,
    GeometryMaterializationKind, GeometryTopologyHistory, ResolvedGeometryRegions,
};
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

const REQUEST_PREFIX: &[u8] = b"runmat-geometry-materialization-request\0";
const REQUEST_DOMAIN: &str = "geometry-materialization-request";
const OUTPUT_PREFIX: &[u8] = b"runmat-geometry-materialization-output\0";
const OUTPUT_DOMAIN: &str = "geometry-materialization-output";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 100_000, 64);
const MAX_SOURCE_CHUNKS: usize = 1_000_000;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometrySourceChunk {
    pub ordinal: u32,
    pub digest: Digest,
    pub encoded_length: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GeometryMaterializationSource {
    Constructed,
    Imported {
        source_name: String,
        source_digest: Digest,
        encoded_length: u64,
        chunks: Vec<GeometrySourceChunk>,
    },
    Derived {
        definition: GeometryDefinitionRef,
        document: Box<GeometryDocument>,
        root: GeometryObjectRef,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryProviderProfile {
    pub provider_id: String,
    pub provider_version: String,
    pub kernel_abi: String,
    pub capabilities: GeometryFeatureCapabilities,
}

impl GeometryProviderProfile {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        for (label, value) in [
            ("provider id", self.provider_id.as_str()),
            ("provider version", self.provider_version.as_str()),
            ("kernel ABI", self.kernel_abi.as_str()),
        ] {
            if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
                return Err(AnalysisBridgeError::Invalid(format!(
                    "geometry {label} is invalid"
                )));
            }
        }
        if self.capabilities.maximum_features == 0
            || self.capabilities.maximum_instances_per_feature == 0
            || self.capabilities.maximum_boolean_operands < 2
        {
            return Err(AnalysisBridgeError::Invalid(
                "geometry provider bounds are invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryMaterializationRequest {
    pub schema_version: u16,
    pub definition: GeometryDefinition,
    pub bindings: BTreeMap<String, GeometryAuthoringValue>,
    pub source: GeometryMaterializationSource,
    pub import_options: ExactCadImportOptions,
    pub provider: GeometryProviderProfile,
    pub plan_limits: GeometryFeaturePlanLimits,
    pub resources: GeometryMaterializationResources,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryMaterializationResources {
    pub cpu_millicores: u32,
    pub maximum_memory_bytes: u64,
    pub maximum_scratch_bytes: u64,
    pub maximum_wall_millis: u64,
    pub maximum_object_bytes: u64,
    pub maximum_artifact_bytes: u64,
    pub maximum_egress_bytes: u64,
    pub maximum_relay_bytes: u64,
}

impl Default for GeometryMaterializationResources {
    fn default() -> Self {
        Self {
            cpu_millicores: 1_000,
            maximum_memory_bytes: 1024 * 1024 * 1024,
            maximum_scratch_bytes: 1024 * 1024 * 1024,
            maximum_wall_millis: 30 * 60 * 1_000,
            maximum_object_bytes: 512 * 1024 * 1024,
            maximum_artifact_bytes: 4 * 1024 * 1024 * 1024,
            maximum_egress_bytes: 4 * 1024 * 1024 * 1024,
            maximum_relay_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl GeometryMaterializationResources {
    pub fn validate(&self, import: &ExactCadImportOptions) -> AnalysisBridgeResult<()> {
        if self.cpu_millicores == 0
            || self.maximum_memory_bytes < import.max_validation_allocation_bytes
            || self.maximum_wall_millis == 0
            || self.maximum_object_bytes < import.max_representation_bytes
            || self.maximum_artifact_bytes < self.maximum_object_bytes
            || self.maximum_egress_bytes < self.maximum_artifact_bytes
        {
            return Err(AnalysisBridgeError::Invalid(
                "geometry materialization resources do not cover the admitted kernel and artifact bounds"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl GeometryMaterializationRequest {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        if self.schema_version != 1
            || self.plan_limits.maximum_features == 0
            || self.plan_limits.maximum_dependency_edges == 0
        {
            return Err(AnalysisBridgeError::Invalid(
                "geometry materialization request has invalid limits or schema".into(),
            ));
        }
        self.definition
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        self.import_options
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        self.provider.validate()?;
        self.resources.validate(&self.import_options)?;
        validate_source(self)?;
        Ok(())
    }

    pub fn identity(&self) -> AnalysisBridgeResult<Digest> {
        Ok(Digest::sha256(&self.canonical_bytes()?))
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            REQUEST_PREFIX,
            REQUEST_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisBridgeResult<Self> {
        let request = decode_contract(REQUEST_PREFIX, REQUEST_DOMAIN, bytes, CONTRACT_LIMITS)?;
        Self::validate(&request)?;
        Ok(request)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryMaterializationOutput {
    pub schema_version: u16,
    pub request_identity: Digest,
    pub definition: GeometryDefinitionRef,
    pub materialization_kind: GeometryMaterializationKind,
    pub provider: GeometryProviderProfile,
    pub document: GeometryDocument,
    pub geometry_root: runmat_execution::value::ValueRef,
    pub regions: ResolvedGeometryRegions,
    pub feature_history: BTreeMap<String, GeometryTopologyHistory>,
    pub feature_resources: Option<GeometryFeatureResourceBounds>,
}

impl GeometryMaterializationOutput {
    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        self.document
            .validate()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        self.provider.validate()?;
        if self.schema_version != 1
            || self.definition.module_path.is_empty()
            || self.definition.geometry_id.is_empty()
            || self.geometry_root.kind != runmat_execution::value::ValueRefKind::ResultObject
            || self.geometry_root.logical_digest.bytes()
                != self.document.primary_artifact().digest.bytes()
            || self.geometry_root.encoded_length != self.document.primary_artifact().encoded_length
            || self.feature_resources.is_some_and(|resources| {
                resources.feature_count == 0
                    || resources.provider_operation_count == 0
                    || resources.maximum_live_artifacts == 0
            })
        {
            return Err(AnalysisBridgeError::Invalid(
                "geometry materialization output is malformed".into(),
            ));
        }
        validate_output_sets(self)
    }

    pub fn validate_against(
        &self,
        request: &GeometryMaterializationRequest,
    ) -> AnalysisBridgeResult<()> {
        request.validate()?;
        self.validate()?;
        let expected_definition = GeometryDefinitionRef {
            module_path: request.definition.module_path.clone(),
            geometry_id: request.definition.geometry_id.clone(),
        };
        if self.schema_version != 1
            || self.request_identity != request.identity()?
            || self.definition != expected_definition
            || self.provider != request.provider
            || self.geometry_root.kind != runmat_execution::value::ValueRefKind::ResultObject
            || self.geometry_root.logical_digest.bytes()
                != self.document.primary_artifact().digest.bytes()
            || self.geometry_root.encoded_length != self.document.primary_artifact().encoded_length
        {
            return Err(AnalysisBridgeError::Identity(
                "geometry materialization output differs from its request or exact document",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            OUTPUT_PREFIX,
            OUTPUT_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisBridgeResult<Self> {
        let output = decode_contract(OUTPUT_PREFIX, OUTPUT_DOMAIN, bytes, CONTRACT_LIMITS)?;
        Self::validate(&output)?;
        Ok(output)
    }
}

fn validate_source(request: &GeometryMaterializationRequest) -> AnalysisBridgeResult<()> {
    use runmat_geometry_core::GeometryDefinitionSource as Definition;
    let valid = match (&request.definition.source, &request.source) {
        (Definition::Constructed(_), GeometryMaterializationSource::Constructed) => true,
        (
            Definition::Imported(imported),
            GeometryMaterializationSource::Imported {
                source_name,
                source_digest,
                encoded_length,
                chunks,
            },
        ) => {
            !source_name.is_empty()
                && source_name.len() <= 4096
                && !source_name.chars().any(char::is_control)
                && source_digest.bytes() == imported.content_digest.bytes()
                && *encoded_length > 0
                && valid_chunks(chunks, *encoded_length, *source_digest)
                && request.import_options.source_units == imported.units
        }
        (
            Definition::Derived(derived),
            GeometryMaterializationSource::Derived {
                definition,
                document,
                root,
            },
        ) => {
            document.validate().is_ok()
                && definition == &derived.source
                && root == document.primary_artifact()
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(AnalysisBridgeError::Invalid(
            "geometry materialization source does not match its definition".into(),
        ))
    }
}

fn valid_chunks(chunks: &[GeometrySourceChunk], encoded_length: u64, digest: Digest) -> bool {
    if chunks.is_empty() || chunks.len() > MAX_SOURCE_CHUNKS {
        return false;
    }
    let mut total = 0_u64;
    for (expected, chunk) in chunks.iter().enumerate() {
        if chunk.ordinal != expected as u32
            || chunk.encoded_length == 0
            || chunk.digest.bytes().iter().all(|byte| *byte == 0)
        {
            return false;
        }
        let Some(next) = total.checked_add(chunk.encoded_length) else {
            return false;
        };
        total = next;
    }
    total == encoded_length && !digest.bytes().iter().all(|byte| *byte == 0)
}

fn validate_output_sets(output: &GeometryMaterializationOutput) -> AnalysisBridgeResult<()> {
    let topology_ids = match &output.document.model {
        GeometryModel::ExactBRep { model } => model.solid_count,
        GeometryModel::FacetedSolid { .. } => 0,
    };
    if topology_ids == 0
        || output
            .feature_history
            .values()
            .any(|history| history.validate().is_err())
    {
        return Err(AnalysisBridgeError::Invalid(
            "geometry materialization output has invalid topology evidence".into(),
        ));
    }
    Ok(())
}

pub fn geometry_digest(value: Digest) -> GeometryDigest {
    GeometryDigest::from_bytes(*value.bytes())
}
