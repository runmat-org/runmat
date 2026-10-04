use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, FieldLocation, IndependentDomain, ObservationComponent,
    ObservationCostEstimate, ObservationId, ObservationQuantity, ObservationSpec,
    ObservationValueShape, PartialAccessPolicy, ProcedureStepId, SamplingPolicy, SignalPrecision,
    SpatialReduction,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

use super::distributed::{
    ObservationPartitionCombination, ObservationPartitionEvidence,
    ObservationPartitionEvidenceIdentity,
};
use super::{
    ObservationAccelerationSummary, ObservationFieldSelectionIdentity, ObservationFieldSource,
};
use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::execution::{ObservationEvaluationRequestIdentity, StepIdentity};
use crate::results::field::{
    RecoveredFieldDefinitionIdentity, RecoveredFieldEntityDomain, RecoveredFieldSample,
};

pub(super) const OBSERVATION_PREFIX: &[u8] = b"runmat-fea-observation/v1\0";
// Artifact provenance was replaced, not made backward compatible. Plan/sample codecs remain
// unchanged because their numerical contracts did not change.
const OBSERVATION_ARTIFACT_PREFIX: &[u8] = b"runmat-fea-observation-artifact/v2\0";
pub(crate) const OBSERVATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResolvedObservationPlanIdentity(pub(super) Digest);

impl ResolvedObservationPlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationResultIdentity(pub(super) Digest);

impl ObservationResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationArtifactIdentity(pub(super) Digest);

impl ObservationArtifactIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationLimits {
    pub maximum_observations: usize,
    pub maximum_selected_entities: usize,
    pub maximum_values: usize,
    pub maximum_operation_count: u64,
    pub maximum_peak_resident_bytes: u64,
    pub maximum_stored_bytes: u64,
    pub maximum_transfer_bytes: u64,
}

impl Default for ObservationLimits {
    fn default() -> Self {
        Self {
            maximum_observations: 4096,
            maximum_selected_entities: 100_000_000,
            maximum_values: 400_000_000,
            maximum_operation_count: 10_000_000_000,
            maximum_peak_resident_bytes: 32 * 1024 * 1024 * 1024,
            maximum_stored_bytes: 32 * 1024 * 1024 * 1024,
            maximum_transfer_bytes: 32 * 1024 * 1024 * 1024,
        }
    }
}

pub(super) use super::entity::observation_element_entity;
pub use super::entity::ObservationEntity;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedObservationTarget {
    Nodes {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Elements {
        domain: RecoveredFieldEntityDomain,
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Points {
        samples: Vec<ResolvedObservationPoint>,
    },
    Volume {
        element_indices: Vec<usize>,
        element_identities: Vec<StableDigest>,
    },
    Boundary {
        face_indices: Vec<usize>,
        face_identities: Vec<StableDigest>,
    },
    Connectors {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    ConstraintGroups {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
}

impl ResolvedObservationTarget {
    pub(crate) fn selected_entity_count(&self) -> usize {
        match self {
            Self::Nodes { indices, .. } | Self::Elements { indices, .. } => indices.len(),
            Self::Points { samples } => samples.len(),
            Self::Volume {
                element_indices, ..
            } => element_indices.len(),
            Self::Boundary { face_indices, .. } => face_indices.len(),
            Self::Connectors { indices, .. } => indices.len(),
            Self::ConstraintGroups { indices, .. } => indices.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservationPoint {
    pub identity: StableDigest,
    pub ordinal: u64,
    pub position_m: [f64; 3],
    pub source_element_index: usize,
    pub source_element_identity: StableDigest,
    pub source_node_indices: Vec<usize>,
    pub source_node_identities: Vec<StableDigest>,
    pub shape_values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservation {
    pub spec: ObservationSpec,
    pub source_field_definition_identity: RecoveredFieldDefinitionIdentity,
    pub source_entity_domain: RecoveredFieldEntityDomain,
    pub target: ResolvedObservationTarget,
    pub component: ResolvedComponentProjection,
    pub result_value_shape: ObservationValueShape,
    pub sample_bounds: ObservationSampleBounds,
    pub resolved_cost: ObservationCostEstimate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationSampleBounds {
    pub maximum_produced_samples: usize,
    pub maximum_retained_samples: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedComponentProjection {
    All,
    Cartesian { axis: usize },
    Magnitude3,
    VonMisesSymmetricTensor,
    MaximumPrincipalSymmetricTensor,
    MinimumPrincipalSymmetricTensor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ResolvedObservationPlanWire")]
pub struct ResolvedObservationPlan {
    pub schema_version: u16,
    pub identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub step_id: ProcedureStepId,
    pub independent_domain: IndependentDomain,
    pub observations: Vec<ResolvedObservation>,
}

impl ResolvedObservationPlan {
    pub(super) fn derive_identity(
        &self,
    ) -> Result<ResolvedObservationPlanIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            &(
                self.schema_version,
                self.topology_digest,
                &self.step_id,
                &self.independent_domain,
                &self.observations,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ResolvedObservationPlanIdentity(Digest::sha256(bytes)))
    }

    pub fn validate_identity(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "observation plan identity does not converge",
            ));
        }
        let mut observation_ids = std::collections::BTreeSet::new();
        for observation in &self.observations {
            if !observation_ids.insert(observation.spec.observation_id.0.as_str()) {
                return Err(ObservationError::Invalid(
                    "resolved observation identifiers must be unique".into(),
                ));
            }
            if !runmat_analysis_core::observation_sampling_matches_domain(
                &self.independent_domain,
                &observation.spec.sampling,
            ) {
                return Err(ObservationError::Invalid(
                    "resolved observation sampling differs from the step domain".into(),
                ));
            }
            validate_resolved_observation(self.topology_digest, observation)?;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate_identity()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let plan = decode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Ok(plan)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedObservationPlanWire {
    schema_version: u16,
    identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    step_id: ProcedureStepId,
    independent_domain: IndependentDomain,
    observations: Vec<ResolvedObservation>,
}

impl TryFrom<ResolvedObservationPlanWire> for ResolvedObservationPlan {
    type Error = ObservationError;

    fn try_from(wire: ResolvedObservationPlanWire) -> Result<Self, Self::Error> {
        let plan = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            topology_digest: wire.topology_digest,
            step_id: wire.step_id,
            independent_domain: wire.independent_domain,
            observations: wire.observations,
        };
        plan.validate_identity()?;
        Ok(plan)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationResultWire")]
pub struct ObservationResult {
    pub schema_version: u16,
    pub logical_identity: ObservationResultIdentity,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub observation_id: ObservationId,
    pub sample: RecoveredFieldSample,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub complex_representation: ComplexRepresentation,
    pub entities: Vec<ObservationEntity>,
    pub field: AnalysisField,
    pub partial_access: PartialAccessPolicy,
    pub partition_evidence: ObservationPartitionEvidence,
}

mod result;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationResultWire {
    schema_version: u16,
    logical_identity: ObservationResultIdentity,
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    observation_id: ObservationId,
    sample: RecoveredFieldSample,
    quantity: ObservationQuantity,
    location: FieldLocation,
    value_shape: ObservationValueShape,
    complex_representation: ComplexRepresentation,
    entities: Vec<ObservationEntity>,
    field: AnalysisField,
    partial_access: PartialAccessPolicy,
    partition_evidence: ObservationPartitionEvidence,
}

impl TryFrom<ObservationResultWire> for ObservationResult {
    type Error = ObservationError;

    fn try_from(wire: ObservationResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            logical_identity: wire.logical_identity,
            plan_identity: wire.plan_identity,
            topology_digest: wire.topology_digest,
            observation_id: wire.observation_id,
            sample: wire.sample,
            quantity: wire.quantity,
            location: wire.location,
            value_shape: wire.value_shape,
            complex_representation: wire.complex_representation,
            entities: wire.entities,
            field: wire.field,
            partial_access: wire.partial_access,
            partition_evidence: wire.partition_evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

mod artifact;
pub use artifact::{
    ObservationArtifact, ObservationArtifactEntry, ObservationArtifactInput,
    ObservationArtifactManifest,
};

mod validation;
use validation::validate_resolved_observation;
pub(super) use validation::{
    derive_resolved_point_identity, materialized_field_location, validate_output_numeric_contract,
};

#[derive(Debug, thiserror::Error)]
pub enum ObservationError {
    #[error("invalid observation contract: {0}")]
    Invalid(String),
    #[error("unsupported observation contract: {0}")]
    Unsupported(&'static str),
    #[error("observation `{observation_id}` cannot resolve target: {reason}")]
    Target {
        observation_id: String,
        reason: String,
    },
    #[error("observation `{observation_id}` cannot bind a recovered field: {reason}")]
    Field {
        observation_id: String,
        reason: String,
    },
    #[error("observation resource limit exceeded: {0}")]
    Limit(&'static str),
    #[error("observation identity mismatch: {0}")]
    Identity(&'static str),
    #[error("observation canonical encoding failed: {0}")]
    Encoding(#[from] runmat_canonical_codec::CanonicalCodecError),
    #[error("observation publication failed: {0}")]
    Publication(#[from] AnalysisExecutionError),
    #[error("analysis topology traversal failed: {0}")]
    Topology(#[from] crate::topology::AnalysisProjectionError),
}

pub(super) fn checked_bytes(values: usize) -> Result<u64, ObservationError> {
    u64::try_from(values)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or(ObservationError::Limit("value byte count"))
}

pub(super) fn observation_result_storage_component_count(
    logical_component_count: usize,
    representation: ComplexRepresentation,
) -> Result<usize, ObservationError> {
    match representation {
        ComplexRepresentation::Complex => {
            logical_component_count
                .checked_mul(2)
                .ok_or(ObservationError::Limit(
                    "complex observation component count",
                ))
        }
        _ => Ok(logical_component_count),
    }
}

pub(super) fn observation_result_storage_shape(
    entity_count: Option<usize>,
    value_shape: &ObservationValueShape,
    representation: ComplexRepresentation,
) -> Result<Vec<usize>, ObservationError> {
    let logical_components = shape_component_count(value_shape)?;
    Ok(match (entity_count, logical_components, representation) {
        (Some(entities), 1, ComplexRepresentation::Complex) => vec![entities, 2],
        (Some(entities), components, ComplexRepresentation::Complex) => {
            vec![entities, components, 2]
        }
        (Some(entities), 1, _) => vec![entities],
        (Some(entities), components, _) => vec![entities, components],
        (None, _, ComplexRepresentation::Complex) => vec![logical_components, 2],
        (None, _, _) => vec![logical_components],
    })
}

pub(super) fn shape_component_count(
    shape: &ObservationValueShape,
) -> Result<usize, ObservationError> {
    shape.component_count().ok_or_else(|| {
        ObservationError::Invalid(
            "tensor observation shapes require positive nonempty extents".into(),
        )
    })
}

pub(super) fn publication_result<T>(
    value: AnalysisExecutionResult<T>,
) -> Result<T, ObservationError> {
    value.map_err(ObservationError::Publication)
}
