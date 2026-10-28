use super::*;

pub fn structural_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<Vec<usize>, ObservationError> {
    let topology = ObservationTopology::derive(model, mesh)?;
    observation_partition_source_indices(plan, partition_plan, partition, &topology)
}

pub fn solid_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
) -> Result<Vec<usize>, ObservationError> {
    SolidObservationSourceSelection::new(plan, projection)?
        .partition_source_indices(partition_plan, partition)
}

/// Source-index selection for every partition of one resolved plan over a solid or volume
/// projection. The observation topology is derived once here and shared, read-only, by every
/// partition of every observation of the plan (including partitions selected concurrently).
pub struct SolidObservationSourceSelection<'a> {
    plan: &'a ResolvedObservationPlan,
    topology: ObservationTopology<'a>,
}

impl<'a> SolidObservationSourceSelection<'a> {
    pub fn new(
        plan: &'a ResolvedObservationPlan,
        projection: &'a AnalysisProjection,
    ) -> Result<Self, ObservationError> {
        Ok(Self {
            plan,
            topology: solid_observation_topology(plan, projection)?,
        })
    }

    pub fn partition_source_indices(
        &self,
        partition_plan: &ObservationPartitionPlan,
        partition: &ObservationPartition,
    ) -> Result<Vec<usize>, ObservationError> {
        observation_partition_source_indices(self.plan, partition_plan, partition, &self.topology)
    }
}

/// The observation topology every partition of `plan` over a solid or volume projection
/// selects from; derive it once per plan, not once per partition.
pub(in super::super) fn solid_observation_topology<'a>(
    plan: &ResolvedObservationPlan,
    projection: &'a AnalysisProjection,
) -> Result<ObservationTopology<'a>, ObservationError> {
    ObservationTopology::from_projection_for(projection, observation_topology_faces(plan))
}

/// Cell- and face-located observations address the mesh's owner-oriented faces too; the
/// solid observation topology of a plan depends on the projection and this flag only.
pub(in super::super) fn observation_topology_faces(plan: &ResolvedObservationPlan) -> bool {
    plan.observations.iter().any(|observation| {
        matches!(
            observation.spec.source_location,
            runmat_analysis_core::FieldLocation::Cell | runmat_analysis_core::FieldLocation::Face
        )
    })
}

pub(in super::super) fn observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    topology: &ObservationTopology,
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
    let topology = ObservationTopology::derive(model, mesh)?;
    prepare_observation_field_partition(plan, partition_plan, partition, field, &topology)
}

pub fn prepare_solid_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    projection: &AnalysisProjection,
    field: &RecoveredField,
) -> Result<RecoveredFieldPartitionSet, ObservationError> {
    let topology = ObservationTopology::from_projection_for(
        projection,
        ObservationTopology::needs_faces(std::iter::once(field.definition())),
    )?;
    prepare_observation_field_partition(plan, partition_plan, partition, field, &topology)
}

pub(in super::super) fn prepare_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    field: &RecoveredField,
    topology: &ObservationTopology,
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
    let field_error = |error: crate::results::field::RecoveredFieldError| ObservationError::Field {
        observation_id: observation.spec.observation_id.0.clone(),
        reason: error.to_string(),
    };
    let support = match topology.support_topology() {
        Some(support_topology) => field
            .definition()
            .support()
            .resolve(field.definition().entity_domain(), support_topology)
            .map_err(field_error)?,
        None if field.definition().support().is_complete() => None,
        None => {
            return Err(ObservationError::Identity(
                "a region-supported field needs a mesh projection topology",
            ))
        }
    };
    let partition = match support {
        Some(support) => RecoveredFieldPartition::select_supported(field, indices, &support),
        None => RecoveredFieldPartition::select(field, indices),
    }
    .map_err(field_error)?;
    RecoveredFieldPartitionSet::new(vec![partition]).map_err(|error| ObservationError::Field {
        observation_id: observation.spec.observation_id.0.clone(),
        reason: error.to_string(),
    })
}
