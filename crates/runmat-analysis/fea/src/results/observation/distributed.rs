use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisField, AnalysisModel, ObservationId, ObservationValueShape, SamplingPolicy,
    SpatialReduction,
};
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
        OBSERVATION_LIMITS, OBSERVATION_PREFIX,
    },
    distributed_numeric::{
        reduction_operation, required_source_indices, select_values,
        weighted_boundary_contribution, weighted_volume_contribution, CompensatedSum,
    },
    topology::StructuralObservationTopology,
    ObservationEntity, ObservationError, ObservationEventAttribution, ObservationExecutionPolicy,
    ObservationPartition, ObservationPartitionIdentity, ObservationPartitionPlan,
    ObservationPartitionPlanIdentity, ObservationPartitionTarget, ObservationResult,
    ObservationResultIdentity, ResolvedObservation, ResolvedObservationPlan,
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
    WeightedReduction {
        linear_sum: Vec<f64>,
        square_sum: Vec<f64>,
        measure: f64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPartitionCombination {
    CanonicalConcatenation,
    CanonicalCompensatedReduction { collective_count: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationPartitionEvidenceWire")]
pub struct ObservationPartitionEvidence {
    pub schema_version: u16,
    pub identity: ObservationPartitionEvidenceIdentity,
    pub partition_plan: ObservationPartitionPlanIdentity,
    pub source_field: RecoveredFieldIdentity,
    pub sample: RecoveredFieldSample,
    pub source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    pub contributions: Vec<ObservationPartitionContributionIdentity>,
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
    source_partitions: Vec<RecoveredFieldPartitionIdentity>,
    contributions: Vec<ObservationPartitionContributionIdentity>,
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
            source_partitions: wire.source_partitions,
            contributions: wire.contributions,
            combination: wire.combination,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl ObservationPartitionEvidence {
    fn new(
        partition_plan: ObservationPartitionPlanIdentity,
        source_field: RecoveredFieldIdentity,
        sample: RecoveredFieldSample,
        source_partitions: Vec<RecoveredFieldPartitionIdentity>,
        contributions: Vec<ObservationPartitionContributionIdentity>,
        combination: ObservationPartitionCombination,
    ) -> Result<Self, ObservationError> {
        let mut evidence = Self {
            schema_version: 1,
            identity: ObservationPartitionEvidenceIdentity(Digest::sha256([])),
            partition_plan,
            source_field,
            sample,
            source_partitions,
            contributions,
            combination,
        };
        evidence.identity = evidence.derive_identity()?;
        evidence.validate()?;
        Ok(evidence)
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
        let combination_is_valid = match self.combination {
            ObservationPartitionCombination::CanonicalConcatenation => true,
            ObservationPartitionCombination::CanonicalCompensatedReduction { collective_count } => {
                collective_count as usize == self.contributions.len().saturating_sub(1)
            }
        };
        if self.schema_version != 1
            || self.source_partitions.is_empty()
            || !source_partitions_are_unique
            || !contributions_are_unique
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
                &self.source_partitions,
                &self.contributions,
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
            payload: wire.payload,
        };
        contribution.validate()?;
        Ok(contribution)
    }
}

impl ObservationPartitionContribution {
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
            ObservationPartitionPayload::SelectedValues { entities, values } => {
                !entities.is_empty()
                    && !values.is_empty()
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
        };
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
    topology: &StructuralObservationTopology,
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
) -> Result<ObservationPartitionContribution, ObservationError> {
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation partition topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition(plan, partition_plan, partition, field, &topology)
}

pub fn evaluate_solid_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
    field: &RecoveredFieldPartitionSet,
) -> Result<ObservationPartitionContribution, ObservationError> {
    let topology = StructuralObservationTopology::from_projection(projection);
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation partition topology differs from its resolved plan",
        ));
    }
    evaluate_observation_partition(plan, partition_plan, partition, field, &topology)
}

pub fn reduce_observation_partitions(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    contributions: Vec<ObservationPartitionContribution>,
) -> Result<ObservationResult, ObservationError> {
    combine_observation_partitions(plan, partition_plan, contributions)
}

pub fn structural_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<Vec<usize>, ObservationError> {
    partition_plan.validate_for(plan)?;
    if partition_plan
        .partitions
        .get(partition.ordinal as usize)
        .is_none_or(|expected| expected != partition)
    {
        return Err(ObservationError::Identity(
            "observation source-index request differs from canonical ownership",
        ));
    }
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("partitioned observation is absent from its plan".into())
        })?;
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation source-index topology differs from its resolved plan",
        ));
    }
    required_source_indices(observation, partition, &topology)
}

pub fn prepare_structural_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    field: &RecoveredField,
) -> Result<RecoveredFieldPartitionSet, ObservationError> {
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    prepare_observation_field_partition(plan, partition_plan, partition, field, &topology)
}

pub fn prepare_solid_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
    field: &RecoveredField,
) -> Result<RecoveredFieldPartitionSet, ObservationError> {
    let topology = StructuralObservationTopology::from_projection(projection);
    prepare_observation_field_partition(plan, partition_plan, partition, field, &topology)
}

fn prepare_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredField,
    topology: &StructuralObservationTopology,
) -> Result<RecoveredFieldPartitionSet, ObservationError> {
    partition_plan.validate_for(plan)?;
    if partition_plan
        .partitions
        .get(partition.ordinal as usize)
        .is_none_or(|expected| expected != partition)
    {
        return Err(ObservationError::Identity(
            "observation field partition request differs from canonical ownership",
        ));
    }
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("partitioned observation is absent from its plan".into())
        })?;
    if field.definition().identity() != observation.source_field_definition_identity
        || topology.digest() != plan.topology_digest
    {
        return Err(ObservationError::Identity(
            "observation field partition source differs from its plan",
        ));
    }
    let indices = required_source_indices(observation, partition, topology)?;
    let partition = RecoveredFieldPartition::select(field, indices).map_err(|error| {
        ObservationError::Field {
            observation_id: observation.spec.observation_id.0.clone(),
            reason: error.to_string(),
        }
    })?;
    RecoveredFieldPartitionSet::new(vec![partition]).map_err(|error| ObservationError::Field {
        observation_id: observation.spec.observation_id.0.clone(),
        reason: error.to_string(),
    })
}

pub(crate) fn evaluate_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    topology: &StructuralObservationTopology,
) -> Result<ObservationPartitionContribution, ObservationError> {
    evaluate_observation_partition_with_attribution(
        plan,
        partition_plan,
        partition,
        field,
        topology,
        None,
    )
}

fn evaluate_observation_partition_with_attribution(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredFieldPartitionSet,
    topology: &StructuralObservationTopology,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationPartitionContribution, ObservationError> {
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
    let sample_matches = if observation.spec.sampling == SamplingPolicy::Final {
        event_attribution.is_none() && field.sample == RecoveredFieldSample::Final
    } else {
        super::history::sample_matches_history_policy(
            &observation.spec.sampling,
            field.sample,
            event_attribution,
        )? && super::history::sample_matches_domain(&plan.independent_domain, field.sample)
    };
    if !sample_matches {
        return Err(ObservationError::Invalid(
            "observation partition source sample differs from its resolved schedule".into(),
        ));
    }
    let payload = match &partition.target {
        ObservationPartitionTarget::Nodes {
            indices,
            identities,
        } => ObservationPartitionPayload::SelectedValues {
            entities: identities
                .iter()
                .copied()
                .map(ObservationEntity::Node)
                .collect(),
            values: select_values(field, indices, observation)?,
        },
        ObservationPartitionTarget::Elements {
            indices,
            identities,
        } => ObservationPartitionPayload::SelectedValues {
            entities: identities
                .iter()
                .copied()
                .map(ObservationEntity::Element)
                .collect(),
            values: select_values(field, indices, observation)?,
        },
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
    };
    let mut contribution = ObservationPartitionContribution {
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
        payload,
    };
    contribution.identity = contribution.derive_identity()?;
    contribution.validate()?;
    Ok(contribution)
}

pub(crate) fn combine_observation_partitions(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    mut contributions: Vec<ObservationPartitionContribution>,
) -> Result<ObservationResult, ObservationError> {
    partition_plan.validate_for(plan)?;
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("partitioned observation is absent from its plan".into())
        })?;
    contributions.sort_by_key(|contribution| contribution.partition_ordinal);
    if contributions.len() != partition_plan.partitions.len() {
        return Err(ObservationError::Identity(
            "observation reduction is missing partition contributions",
        ));
    }
    for (partition, contribution) in partition_plan.partitions.iter().zip(&contributions) {
        contribution.validate()?;
        if contribution.partition_plan != partition_plan.identity
            || contribution.partition != partition.identity
            || contribution.partition_ordinal != partition.ordinal
            || contribution.observation_id != partition_plan.observation_id
            || contribution.source_field_definition != observation.source_field_definition_identity
        {
            return Err(ObservationError::Identity(
                "observation contribution differs from its partition plan",
            ));
        }
        validate_contribution_payload(partition, contribution, observation)?;
    }
    let first_source = contributions[0].source_field;
    let first_sample = contributions[0].sample;
    if contributions.iter().any(|contribution| {
        contribution.source_field != first_source || contribution.sample != first_sample
    }) {
        return Err(ObservationError::Identity(
            "observation contributions name different source fields or samples",
        ));
    }
    let source_partitions = contributions
        .iter()
        .flat_map(|contribution| contribution.source_partitions.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let contribution_identities = contributions
        .iter()
        .map(|contribution| contribution.identity)
        .collect::<Vec<_>>();
    let combination = match &contributions[0].payload {
        ObservationPartitionPayload::SelectedValues { .. } => {
            ObservationPartitionCombination::CanonicalConcatenation
        }
        ObservationPartitionPayload::WeightedReduction { .. } => {
            ObservationPartitionCombination::CanonicalCompensatedReduction {
                collective_count: u32::try_from(contributions.len().saturating_sub(1))
                    .map_err(|_| ObservationError::Limit("observation collective count"))?,
            }
        }
    };
    let (values, entities) = match &contributions[0].payload {
        ObservationPartitionPayload::SelectedValues { .. } => {
            let mut entities = Vec::new();
            let mut values = Vec::new();
            for contribution in contributions {
                let ObservationPartitionPayload::SelectedValues {
                    entities: local_entities,
                    values: local_values,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                entities.extend(local_entities);
                values.extend(local_values);
            }
            (values, entities)
        }
        ObservationPartitionPayload::WeightedReduction { .. } => {
            let operation = reduction_operation(observation)?;
            let component_count = observation_result_storage_component_count(
                observation.result_component_count,
                observation.spec.complex_representation,
            )?;
            let mut linear = vec![CompensatedSum::default(); component_count];
            let mut square = vec![CompensatedSum::default(); component_count];
            let mut measure = CompensatedSum::default();
            for contribution in contributions {
                let ObservationPartitionPayload::WeightedReduction {
                    linear_sum,
                    square_sum,
                    measure: local_measure,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                if linear_sum.len() != component_count || square_sum.len() != component_count {
                    return Err(ObservationError::Identity(
                        "observation partition component count differs from its plan",
                    ));
                }
                for (total, local) in linear.iter_mut().zip(linear_sum) {
                    total.add(local);
                }
                for (total, local) in square.iter_mut().zip(square_sum) {
                    total.add(local);
                }
                measure.add(local_measure);
            }
            let measure = measure.total();
            if !measure.is_finite() || measure <= 0.0 {
                return Err(ObservationError::Invalid(
                    "reduction target has no positive finite measure".into(),
                ));
            }
            let values = match operation {
                SpatialReduction::Integral => linear.into_iter().map(|sum| sum.total()).collect(),
                SpatialReduction::Average => linear
                    .into_iter()
                    .map(|sum| sum.total() / measure)
                    .collect(),
                SpatialReduction::RootMeanSquare => square
                    .into_iter()
                    .map(|sum| (sum.total() / measure).sqrt())
                    .collect(),
                _ => {
                    return Err(ObservationError::Unsupported(
                        "unsupported partitioned spatial reduction",
                    ))
                }
            };
            (values, Vec::new())
        }
    };
    let components = observation.result_component_count;
    let value_shape = if components == 1 {
        ObservationValueShape::Scalar
    } else {
        observation.spec.value_shape.clone()
    };
    let shape = observation_result_storage_shape(
        (!entities.is_empty()).then_some(entities.len()),
        &value_shape,
        observation.spec.complex_representation,
    )?;
    let partition_evidence = ObservationPartitionEvidence::new(
        partition_plan.identity,
        first_source,
        first_sample,
        source_partitions,
        contribution_identities,
        combination,
    )?;
    let mut result = ObservationResult {
        schema_version: 1,
        logical_identity: ObservationResultIdentity(Digest::sha256([])),
        plan_identity: plan.identity,
        topology_digest: plan.topology_digest,
        observation_id: observation.spec.observation_id.clone(),
        sample: first_sample,
        quantity: observation.spec.quantity,
        location: observation.spec.output_location,
        value_shape,
        complex_representation: observation.spec.complex_representation,
        entities,
        field: AnalysisField::host_f64(observation.spec.observation_id.0.clone(), shape, values),
        partial_access: observation.spec.storage.partial_access,
        partition_evidence,
    };
    result.logical_identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}

fn validate_contribution_payload(
    partition: &ObservationPartition,
    contribution: &ObservationPartitionContribution,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    let storage_components = observation_result_storage_component_count(
        observation.result_component_count,
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
            ObservationPartitionTarget::Elements { identities, .. },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(ObservationEntity::Element)
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
