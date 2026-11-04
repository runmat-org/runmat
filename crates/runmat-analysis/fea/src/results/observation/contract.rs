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
/// Observation plans and samples grow with the observed model. Their size is admitted by the
/// resource plan and checked by the object inventory; the codec bounds them by the wire range.
pub(crate) const OBSERVATION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

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

/// Resolved observation costs are exact functions of the topology and the requests, so no
/// model-size constant bounds them: entity, value and byte ceilings are the wire range. The
/// authored request count is a format bound. A run's resident ceiling comes from its step's
/// resource declaration (`for_step`); stored and transferred bytes are admitted against the
/// output plan by the object inventory.
impl Default for ObservationLimits {
    fn default() -> Self {
        let wire = crate::contracts::canonical_digest::WIRE_BOUND;
        Self {
            maximum_observations: 4096,
            maximum_selected_entities: wire,
            maximum_values: wire,
            maximum_operation_count: wire as u64,
            maximum_peak_resident_bytes: wire as u64,
            maximum_stored_bytes: wire as u64,
            maximum_transfer_bytes: wire as u64,
        }
    }
}

impl ObservationLimits {
    /// Limits for one step: the authored resident ceiling, when declared, bounds the peak
    /// resident bytes of observation resolution and evaluation.
    pub fn for_step(step: &runmat_analysis_core::ProcedureStep) -> Self {
        let defaults = Self::default();
        Self {
            maximum_peak_resident_bytes: step
                .accuracy
                .resources
                .maximum_resident_bytes
                .unwrap_or(defaults.maximum_peak_resident_bytes),
            ..defaults
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
    /// Boundary faces of two disjoint face regions in canonical order: the first region's faces
    /// occupy `[0, first_count)`, the second region's the rest.
    BoundaryPair {
        face_indices: Vec<usize>,
        face_identities: Vec<StableDigest>,
        first_count: usize,
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
            Self::Boundary { face_indices, .. } | Self::BoundaryPair { face_indices, .. } => {
                face_indices.len()
            }
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

/// The content of a resolved observation plan. Building or editing contents grants nothing:
/// only `ResolvedObservationPlan::try_from` (which checks the identity and every observation) or
/// the resolver turns contents into a plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservationPlanContents {
    pub schema_version: u16,
    pub identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub step_id: ProcedureStepId,
    pub independent_domain: IndependentDomain,
    pub observations: Vec<ResolvedObservation>,
}

impl ResolvedObservationPlanContents {
    pub fn derive_identity(&self) -> Result<ResolvedObservationPlanIdentity, ObservationError> {
        Ok(ResolvedObservationPlanIdentity(
            crate::contracts::canonical_digest::canonical_contract_digest(
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
            )?,
        ))
    }

    fn validate(&self) -> Result<(), ObservationError> {
        if self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "observation plan identity does not converge",
            ));
        }
        self.validate_observations()
    }

    /// Everything but the identity hash.
    fn validate_observations(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1 {
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
}

/// A resolved observation plan whose identity, schema, unique identifiers, sampling domain and
/// per-observation contracts were checked exactly once, when it was resolved or decoded. The
/// contents are private and read-only (through `Deref`), so holders never re-validate it: the
/// canonical encode behind the identity runs once per plan, not once per consumer.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "ResolvedObservationPlanContents")]
pub struct ResolvedObservationPlan {
    contents: ResolvedObservationPlanContents,
}

/// Encodes exactly as its contents (the wire form is unchanged).
impl Serialize for ResolvedObservationPlan {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.contents.serialize(serializer)
    }
}

impl std::ops::Deref for ResolvedObservationPlan {
    type Target = ResolvedObservationPlanContents;

    fn deref(&self) -> &Self::Target {
        &self.contents
    }
}

impl ResolvedObservationPlan {
    /// Seal contents whose identity the caller derived from these same contents.
    pub(super) fn seal_derived(
        mut contents: ResolvedObservationPlanContents,
    ) -> Result<Self, ObservationError> {
        contents.identity = contents.derive_identity()?;
        contents.validate_observations()?;
        Ok(Self { contents })
    }

    /// Give up the seal to edit the contents; re-sealing validates them again.
    pub fn into_contents(self) -> ResolvedObservationPlanContents {
        self.contents
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
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

impl TryFrom<ResolvedObservationPlanContents> for ResolvedObservationPlan {
    type Error = ObservationError;

    fn try_from(contents: ResolvedObservationPlanContents) -> Result<Self, Self::Error> {
        contents.validate()?;
        Ok(Self { contents })
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

mod point;
pub(super) use point::{derive_resolved_point_identity, resolved_point_identity_resident_bytes};
mod validation;
use validation::validate_resolved_observation;
pub(super) use validation::{materialized_field_location, validate_output_numeric_contract};

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
