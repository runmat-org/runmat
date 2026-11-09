mod combination;
mod payload_bound;
pub(crate) use combination::combine_observation_partitions;
pub use payload_bound::ObservationPartitionPayloadCapacity;
use std::collections::BTreeSet;

use runmat_analysis_core::{AnalysisField, AnalysisModel, ObservationId, SpatialReduction};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{
    results::field::{
        RecoveredField, RecoveredFieldDefinitionIdentity, RecoveredFieldIdentity,
        RecoveredFieldPartition, RecoveredFieldPartitionIdentity, RecoveredFieldPartitionSet,
        RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::{
    contract::{
        observation_result_storage_component_count, observation_result_storage_shape,
        shape_component_count, OBSERVATION_LIMITS, OBSERVATION_PREFIX,
    },
    distributed_numeric::{
        discrete_reaction_contribution, interpolate_values, paired_boundary_contribution,
        reduction_operation, required_source_indices, select_values,
        weighted_boundary_contribution, weighted_volume_contribution,
    },
    extrema::{
        extrema_values, partition_selected_entities, selected_or_extrema_payload, spatial_extrema,
    },
    topology::ObservationTopology,
    ObservationAccelerationEvidence, ObservationEntity, ObservationError,
    ObservationEventAttribution, ObservationExecutionPolicy, ObservationPartition,
    ObservationPartitionIdentity, ObservationPartitionPlan, ObservationPartitionPlanIdentity,
    ObservationPartitionTarget, ObservationResult, ObservationResultIdentity, ResolvedObservation,
    ResolvedObservationPlan,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationPartitionContributionIdentity(Digest);

impl ObservationPartitionContributionIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationPartitionEvidenceIdentity(Digest);

impl ObservationPartitionEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ObservationPartitionPayload {
    SelectedValues {
        entities: Vec<ObservationEntity>,
        values: Vec<f64>,
    },
    ExtremaCandidates {
        entities: Vec<ObservationEntity>,
        values: Vec<f64>,
    },
    WeightedReduction {
        linear_sum: Vec<f64>,
        square_sum: Vec<f64>,
        measure: f64,
    },
    /// Area integrals and areas of a scalar over the two regions of a paired boundary target.
    /// A partition may hold faces of one region only, so either measure may be zero here; the
    /// combined measures must both be positive.
    PairedWeightedReduction {
        first_sum: f64,
        first_measure: f64,
        second_sum: f64,
        second_measure: f64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPartitionCombination {
    CanonicalConcatenation,
    CanonicalCompensatedReduction { collective_count: u32 },
    CanonicalExtremaReduction { collective_count: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationPartitionEvidenceWire")]
pub struct ObservationPartitionEvidence {
    pub schema_version: u16,
    pub identity: ObservationPartitionEvidenceIdentity,
    pub partition_plan: ObservationPartitionPlanIdentity,
    pub source_field: RecoveredFieldIdentity,
    pub sample: RecoveredFieldSample,
    pub event_attribution: Option<ObservationEventAttribution>,
    pub source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    pub contributions: Vec<ObservationPartitionContributionIdentity>,
    pub accelerations: Vec<ObservationAccelerationEvidence>,
    pub combination: ObservationPartitionCombination,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationPartitionEvidenceWire {
    schema_version: u16,
    identity: ObservationPartitionEvidenceIdentity,
    partition_plan: ObservationPartitionPlanIdentity,
    source_field: RecoveredFieldIdentity,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
    source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    contributions: Vec<ObservationPartitionContributionIdentity>,
    accelerations: Vec<ObservationAccelerationEvidence>,
    combination: ObservationPartitionCombination,
}

struct ObservationPartitionEvidenceInput {
    partition_plan: ObservationPartitionPlanIdentity,
    source_field: RecoveredFieldIdentity,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
    source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    contributions: Vec<ObservationPartitionContributionIdentity>,
    accelerations: Vec<ObservationAccelerationEvidence>,
    combination: ObservationPartitionCombination,
}

impl TryFrom<ObservationPartitionEvidenceWire> for ObservationPartitionEvidence {
    type Error = ObservationError;

    fn try_from(wire: ObservationPartitionEvidenceWire) -> Result<Self, Self::Error> {
        let evidence = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            partition_plan: wire.partition_plan,
            source_field: wire.source_field,
            sample: wire.sample,
            event_attribution: wire.event_attribution,
            source_partitions: wire.source_partitions,
            contributions: wire.contributions,
            accelerations: wire.accelerations,
            combination: wire.combination,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl ObservationPartitionEvidence {
    fn new(input: ObservationPartitionEvidenceInput) -> Result<Self, ObservationError> {
        let mut evidence = Self {
            schema_version: 1,
            identity: ObservationPartitionEvidenceIdentity(Digest::sha256([])),
            partition_plan: input.partition_plan,
            source_field: input.source_field,
            sample: input.sample,
            event_attribution: input.event_attribution,
            source_partitions: input.source_partitions,
            contributions: input.contributions,
            accelerations: input.accelerations,
            combination: input.combination,
        };
        evidence.identity = evidence.derive_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    /// Evidence of an observation evaluated directly from one solver result, as one partition
    /// and one contribution: no recovered field is partitioned or combined.
    pub(super) fn direct(
        partition_plan: ObservationPartitionPlanIdentity,
        source_field: RecoveredFieldIdentity,
        contribution: Digest,
        sample: RecoveredFieldSample,
    ) -> Result<Self, ObservationError> {
        Self::new(ObservationPartitionEvidenceInput {
            partition_plan,
            source_field,
            sample,
            event_attribution: None,
            source_partitions: vec![RecoveredFieldPartitionIdentity::derived(
                source_field.digest(),
            )],
            contributions: vec![ObservationPartitionContributionIdentity(contribution)],
            accelerations: Vec::new(),
            combination: ObservationPartitionCombination::CanonicalCompensatedReduction {
                collective_count: 0,
            },
        })
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let source_partitions_are_unique = self
            .source_partitions
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == self.source_partitions.len();
        let contributions_are_unique = self
            .contributions
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == self.contributions.len();
        let accelerations_are_unique = self
            .accelerations
            .iter()
            .map(|evidence| evidence.identity)
            .collect::<BTreeSet<_>>()
            .len()
            == self.accelerations.len();
        for acceleration in &self.accelerations {
            acceleration.validate()?;
        }
        let combination_is_valid = match self.combination {
            ObservationPartitionCombination::CanonicalConcatenation => true,
            ObservationPartitionCombination::CanonicalCompensatedReduction { collective_count }
            | ObservationPartitionCombination::CanonicalExtremaReduction { collective_count } => {
                collective_count as usize == self.contributions.len().saturating_sub(1)
            }
        };
        if let Some(attribution) = self.event_attribution {
            attribution.validate()?;
            if attribution.sample() != self.sample {
                return Err(ObservationError::Invalid(
                    "observation partition evidence event differs from its source sample".into(),
                ));
            }
        }
        if self.schema_version != 1
            || self.source_partitions.is_empty()
            || !source_partitions_are_unique
            || !contributions_are_unique
            || !accelerations_are_unique
            || !combination_is_valid
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation partition evidence is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ObservationPartitionEvidenceIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-evidence-identity",
            &(
                self.schema_version,
                self.partition_plan,
                self.source_field,
                self.sample,
                self.event_attribution,
                &self.source_partitions,
                &self.contributions,
                &self.accelerations,
                self.combination,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationPartitionEvidenceIdentity(Digest::sha256(bytes)))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationPartitionContributionWire")]
pub struct ObservationPartitionContribution {
    pub schema_version: u16,
    pub identity: ObservationPartitionContributionIdentity,
    pub partition_plan: ObservationPartitionPlanIdentity,
    pub partition: ObservationPartitionIdentity,
    pub partition_ordinal: u32,
    pub observation_id: ObservationId,
    pub source_field_definition: RecoveredFieldDefinitionIdentity,
    pub source_field: RecoveredFieldIdentity,
    pub source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    pub source_entity_count: usize,
    pub sample: RecoveredFieldSample,
    pub event_attribution: Option<ObservationEventAttribution>,
    pub acceleration: Option<ObservationAccelerationEvidence>,
    pub payload: ObservationPartitionPayload,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationPartitionContributionWire {
    schema_version: u16,
    identity: ObservationPartitionContributionIdentity,
    partition_plan: ObservationPartitionPlanIdentity,
    partition: ObservationPartitionIdentity,
    partition_ordinal: u32,
    observation_id: ObservationId,
    source_field_definition: RecoveredFieldDefinitionIdentity,
    source_field: RecoveredFieldIdentity,
    source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    source_entity_count: usize,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
    acceleration: Option<ObservationAccelerationEvidence>,
    payload: ObservationPartitionPayload,
}

impl TryFrom<ObservationPartitionContributionWire> for ObservationPartitionContribution {
    type Error = ObservationError;

    fn try_from(wire: ObservationPartitionContributionWire) -> Result<Self, Self::Error> {
        let contribution = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            partition_plan: wire.partition_plan,
            partition: wire.partition,
            partition_ordinal: wire.partition_ordinal,
            observation_id: wire.observation_id,
            source_field_definition: wire.source_field_definition,
            source_field: wire.source_field,
            source_partitions: wire.source_partitions,
            source_entity_count: wire.source_entity_count,
            sample: wire.sample,
            event_attribution: wire.event_attribution,
            acceleration: wire.acceleration,
            payload: wire.payload,
        };
        contribution.validate()?;
        Ok(contribution)
    }
}

impl ObservationPartitionContribution {
    pub(super) fn new(
        partition_plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
        field: &RecoveredFieldPartitionSet,
        event_attribution: Option<ObservationEventAttribution>,
        acceleration: Option<ObservationAccelerationEvidence>,
        payload: ObservationPartitionPayload,
    ) -> Result<Self, ObservationError> {
        let mut contribution = Self {
            schema_version: 1,
            identity: ObservationPartitionContributionIdentity(Digest::sha256([])),
            partition_plan: partition_plan.identity,
            partition: partition.identity,
            partition_ordinal: partition.ordinal,
            observation_id: partition_plan.observation_id.clone(),
            source_field_definition: field.definition.identity(),
            source_field: field.source_field,
            source_partitions: field
                .partitions
                .iter()
                .map(|partition| partition.identity)
                .collect(),
            source_entity_count: field.source_indices().count(),
            sample: field.sample,
            event_attribution,
            acceleration,
            payload,
        };
        contribution.identity = contribution.derive_identity()?;
        contribution.validate()?;
        Ok(contribution)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-contribution",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        Ok(decode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-contribution",
            bytes,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let payload_is_valid = match &self.payload {
            ObservationPartitionPayload::SelectedValues { entities, values }
            | ObservationPartitionPayload::ExtremaCandidates { entities, values } => {
                !entities.is_empty()
                    && !values.is_empty()
                    && entities.len() <= values.len()
                    && values.iter().all(|value| value.is_finite())
            }
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            } => {
                !linear_sum.is_empty()
                    && linear_sum.len() == square_sum.len()
                    && linear_sum
                        .iter()
                        .chain(square_sum)
                        .all(|value| value.is_finite())
                    && measure.is_finite()
                    && *measure > 0.0
            }
            ObservationPartitionPayload::PairedWeightedReduction {
                first_sum,
                first_measure,
                second_sum,
                second_measure,
            } => {
                [first_sum, first_measure, second_sum, second_measure]
                    .iter()
                    .all(|value| value.is_finite())
                    && *first_measure >= 0.0
                    && *second_measure >= 0.0
                    && first_measure + second_measure > 0.0
            }
        };
        if let Some(attribution) = self.event_attribution {
            attribution.validate()?;
            if attribution.sample() != self.sample {
                return Err(ObservationError::Invalid(
                    "observation partition contribution event differs from its source sample"
                        .into(),
                ));
            }
        }
        if let Some(acceleration) = &self.acceleration {
            acceleration.validate()?;
            if !matches!(
                self.payload,
                ObservationPartitionPayload::ExtremaCandidates { .. }
            ) {
                return Err(ObservationError::Invalid(
                    "accelerated observation contribution must contain extrema candidates".into(),
                ));
            }
        }
        if self.schema_version != 1
            || self.observation_id.0.trim().is_empty()
            || self.source_entity_count == 0
            || self.source_partitions.is_empty()
            || self
                .source_partitions
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != self.source_partitions.len()
            || !payload_is_valid
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation partition contribution metadata, payload, or identity is invalid"
                    .into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(
        &self,
    ) -> Result<ObservationPartitionContributionIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "observation-partition-contribution-identity",
            &(
                self.schema_version,
                self.partition_plan,
                self.partition,
                self.partition_ordinal,
                &self.observation_id,
                self.source_field_definition,
                self.source_field,
                &self.source_partitions,
                self.source_entity_count,
                self.sample,
                self.event_attribution,
                &self.payload,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationPartitionContributionIdentity(Digest::sha256(
            bytes,
        )))
    }
}

pub(crate) fn evaluate_partitioned_observation(
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    field: &RecoveredField,
    topology: &ObservationTopology,
    execution_policy: ObservationExecutionPolicy,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationResult, ObservationError> {
    let partition_plan =
        ObservationPartitionPlan::derive(plan, &observation.spec.observation_id, execution_policy)?;
    let contributions = partition_plan
        .partitions
        .iter()
        .map(|partition| {
            let source = prepare_observation_field_partition(
                plan,
                &partition_plan,
                partition,
                field,
                topology,
            )?;
            evaluate_observation_partition_with_attribution(
                plan,
                &partition_plan,
                partition,
                &source,
                topology,
                event_attribution,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    combine_observation_partitions(plan, &partition_plan, contributions)
}

pub fn evaluate_structural_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let topology = ObservationTopology::derive(model, mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation partition topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition_with_attribution(
        plan,
        partition_plan,
        partition,
        field,
        &topology,
        event_attribution,
    )
}

pub fn evaluate_solid_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let topology = ObservationTopology::from_projection_for(
        projection,
        ObservationTopology::needs_faces(std::iter::once(&field.definition)),
    )?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation partition topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition_with_attribution(
        plan,
        partition_plan,
        partition,
        field,
        &topology,
        event_attribution,
    )
}

pub fn reduce_observation_partitions(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    contributions: Vec<ObservationPartitionContribution>,
) -> Result<ObservationResult, ObservationError> {
    combine_observation_partitions(plan, partition_plan, contributions)
}

pub(super) mod source_selection;
use source_selection::prepare_observation_field_partition;
pub use source_selection::{
    prepare_solid_observation_field_partition, prepare_structural_observation_field_partition,
    solid_observation_partition_source_indices, structural_observation_partition_source_indices,
    SolidObservationSourceSelection,
};

pub(super) fn validate_observation_partition_input<'a>(
    plan: &'a ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<&'a ResolvedObservation, ObservationError> {
    partition_plan.validate_for(plan)?;
    let expected_partition = partition_plan
        .partitions
        .get(partition.ordinal as usize)
        .ok_or(ObservationError::Identity(
            "observation partition is absent from its plan",
        ))?;
    if expected_partition != partition {
        return Err(ObservationError::Identity(
            "observation partition differs from its canonical assignment",
        ));
    }
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("partitioned observation is absent from its plan".into())
        })?;
    field.validate().map_err(|error| ObservationError::Field {
        observation_id: observation.spec.observation_id.0.clone(),
        reason: error.to_string(),
    })?;
    if field.definition.identity() != observation.source_field_definition_identity {
        return Err(ObservationError::Identity(
            "observation partition received a substituted source field",
        ));
    }
    if !super::evaluate::resolved_observation_accepts_sample(
        plan,
        observation,
        field.sample,
        event_attribution,
    )? {
        return Err(ObservationError::Invalid(
            "observation partition source sample differs from its resolved schedule".into(),
        ));
    }
    Ok(observation)
}

fn observation_partition_payload(
    observation: &ResolvedObservation,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    topology: &ObservationTopology,
) -> Result<ObservationPartitionPayload, ObservationError> {
    Ok(match &partition.target {
        ObservationPartitionTarget::Nodes {
            indices,
            identities,
        } => {
            if matches!(
                reduction_operation(observation),
                Ok(SpatialReduction::Resultant | SpatialReduction::Moment { .. })
            ) {
                discrete_reaction_contribution(observation, field, indices, topology)?
            } else {
                selected_or_extrema_payload(
                    observation,
                    identities
                        .iter()
                        .copied()
                        .map(ObservationEntity::Node)
                        .collect(),
                    select_values(field, indices, observation)?,
                )?
            }
        }
        ObservationPartitionTarget::Elements {
            domain,
            indices,
            identities,
        } => selected_or_extrema_payload(
            observation,
            identities
                .iter()
                .copied()
                .map(|identity| super::contract::observation_element_entity(*domain, identity))
                .collect(),
            select_values(field, indices, observation)?,
        )?,
        ObservationPartitionTarget::Connectors {
            indices,
            identities,
        } => {
            let values = select_values(field, indices, observation)?;
            if matches!(
                reduction_operation(observation),
                Ok(SpatialReduction::Resultant)
            ) {
                ObservationPartitionPayload::WeightedReduction {
                    square_sum: vec![0.0; values.len()],
                    linear_sum: values,
                    measure: indices.len() as f64,
                }
            } else {
                selected_or_extrema_payload(
                    observation,
                    identities
                        .iter()
                        .copied()
                        .map(ObservationEntity::Connector)
                        .collect(),
                    values,
                )?
            }
        }
        ObservationPartitionTarget::Points {
            sample_ordinals, ..
        } => {
            let (entities, values) = interpolate_values(field, sample_ordinals, observation)?;
            selected_or_extrema_payload(observation, entities, values)?
        }
        ObservationPartitionTarget::Volume {
            element_indices, ..
        } => weighted_volume_contribution(
            observation,
            field,
            element_indices,
            topology
                .solid_projection
                .as_ref()
                .ok_or(ObservationError::Unsupported(
                    "volume observations require admitted solid topology",
                ))?,
        )?,
        ObservationPartitionTarget::Boundary { face_indices, .. } => {
            weighted_boundary_contribution(
                observation,
                field,
                face_indices,
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "boundary observations require admitted solid topology",
                    ))?,
            )?
        }
        ObservationPartitionTarget::BoundaryPair {
            face_indices,
            first_count,
            ..
        } => {
            paired_boundary_contribution(observation, field, face_indices, *first_count, topology)?
        }
        ObservationPartitionTarget::ConstraintGroups { indices, .. } => {
            let values = select_values(field, indices, observation)?;
            ObservationPartitionPayload::WeightedReduction {
                square_sum: vec![0.0; values.len()],
                linear_sum: values,
                measure: indices.len() as f64,
            }
        }
    })
}

pub(super) fn evaluate_observation_partition_with_attribution(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    topology: &ObservationTopology,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let observation = validate_observation_partition_input(
        plan,
        partition_plan,
        partition,
        field,
        event_attribution,
    )?;
    let payload = observation_partition_payload(observation, partition, field, topology)?;
    ObservationPartitionContribution::new(
        partition_plan,
        partition,
        field,
        event_attribution,
        None,
        payload,
    )
}

fn validate_contribution_payload(
    partition: &ObservationPartition,
    contribution: &ObservationPartitionContribution,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    let storage_components = observation_result_storage_component_count(
        shape_component_count(&observation.result_value_shape)?,
        observation.spec.complex_representation,
    )?;
    match (&partition.target, &contribution.payload) {
        (
            ObservationPartitionTarget::Nodes { identities, .. },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(ObservationEntity::Node)
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "nodal observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Nodes { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant | SpatialReduction::Moment { .. })
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::ConstraintGroups { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant)
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::Elements {
                domain, identities, ..
            },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(|identity| super::contract::observation_element_entity(*domain, identity))
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "element observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Points {
                sample_ordinals,
                identities,
            },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let super::ResolvedObservationTarget::Points { samples } = &observation.target else {
                return Err(ObservationError::Identity(
                    "point contribution does not match its resolved target",
                ));
            };
            let expected = sample_ordinals
                .iter()
                .map(|ordinal| {
                    let sample = samples.get(*ordinal).ok_or(ObservationError::Identity(
                        "point contribution exceeds its resolved target",
                    ))?;
                    Ok(ObservationEntity::Point {
                        identity: sample.identity,
                        ordinal: sample.ordinal,
                        position_m: sample.position_m,
                        source_element: sample.source_element_identity,
                    })
                })
                .collect::<Result<Vec<_>, ObservationError>>()?;
            if identities
                != &expected
                    .iter()
                    .map(|entity| match entity {
                        ObservationEntity::Point { identity, .. } => *identity,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>()
                || entities != &expected
                || values.len()
                    != sample_ordinals
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "point observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Connectors { identities, .. },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(ObservationEntity::Connector)
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "connector observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Connectors { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant)
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::Nodes { .. }
            | ObservationPartitionTarget::Elements { .. }
            | ObservationPartitionTarget::Points { .. },
            ObservationPartitionPayload::ExtremaCandidates { entities, values },
        ) if spatial_extrema(observation).is_some() => {
            let owned = partition_selected_entities(partition, observation)?;
            let mut previous = None;
            for entity in entities {
                let Some(index) = owned.iter().position(|candidate| candidate == entity) else {
                    return Err(ObservationError::Identity(
                        "extrema candidate is not owned by its observation partition",
                    ));
                };
                if previous.is_some_and(|previous| index <= previous) {
                    return Err(ObservationError::Identity(
                        "extrema candidates are not in canonical partition order",
                    ));
                }
                previous = Some(index);
            }
            if entities.is_empty()
                || entities.len() != values.len()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(ObservationError::Identity(
                    "extrema candidate values differ from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::BoundaryPair {
                face_indices,
                first_count,
                ..
            },
            ObservationPartitionPayload::PairedWeightedReduction {
                first_measure,
                second_measure,
                ..
            },
        ) if (*first_count > 0) == (*first_measure > 0.0)
            && (*first_count < face_indices.len()) == (*second_measure > 0.0) => {}
        (
            ObservationPartitionTarget::Volume { .. } | ObservationPartitionTarget::Boundary { .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                ..
            },
        ) if linear_sum.len() == storage_components && square_sum.len() == storage_components => {}
        _ => {
            return Err(ObservationError::Identity(
                "observation contribution payload differs from partition target",
            ))
        }
    }
    Ok(())
}
