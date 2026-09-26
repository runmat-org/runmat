use super::*;

pub fn structural_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<Vec<usize>, ObservationError> {
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    observation_partition_source_indices(plan, partition_plan, partition, &topology)
}

pub fn solid_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
) -> Result<Vec<usize>, ObservationError> {
    let topology = StructuralObservationTopology::from_projection(projection);
    observation_partition_source_indices(plan, partition_plan, partition, &topology)
}

fn observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    topology: &StructuralObservationTopology,
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
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "observation source-index topology differs from its resolved plan",
        ));
    }
    required_source_indices(observation, partition, topology)
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

pub(super) fn prepare_observation_field_partition(
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
