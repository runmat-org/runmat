//! Selection over retained native samples. Never recovers or solves fields.
use super::{
    topology::ObservationTopology, ObservationError, ObservationLimits, ObservationPartition,
    ObservationPartitionContribution, ObservationPartitionPlan, ObservationResult,
    ResolvedObservationPlan,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition, RecoveredFieldPartitionSet,
    SampledVolumeFieldTopology,
};
use runmat_analysis_core::{
    ObservationRequest, ProcedureStep, RecoveryPolicy, SamplingPolicy, SpatialReduction,
};
use runmat_meshing_core::SolverMeshArtifact;

fn validate_spec(spec: &runmat_analysis_core::ObservationSpec) -> Result<(), ObservationError> {
    if spec.sampling != SamplingPolicy::Final
        || spec.recovery != RecoveryPolicy::Native
        || spec.reduction.as_ref().is_some_and(|reduction| {
            !matches!(
                reduction.spatial,
                SpatialReduction::None | SpatialReduction::Minimum | SpatialReduction::Maximum
            )
        })
    {
        return Err(ObservationError::Unsupported("sampled native observations support final fields and scalar extrema, not interpolation or weighted edge circulation"));
    }
    Ok(())
}

pub(crate) fn validate_sampled_requests(step: &ProcedureStep) -> Result<(), ObservationError> {
    for request in &step.observations {
        match request {
            ObservationRequest::Field(spec) | ObservationRequest::Reduction(spec) => {
                validate_spec(spec)?
            }
            _ => {
                return Err(ObservationError::Unsupported(
                    "sampled native histories and probes are not admitted",
                ))
            }
        }
    }
    Ok(())
}

pub fn resolve_sampled_observation_plan(
    step: &ProcedureStep,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    super::resolve::validate_request_counts(step, limits)?;
    validate_sampled_requests(step)?;
    let topology = ObservationTopology::from_sampled(sampling, mesh)?;
    super::resolve::resolve_observation_plan_for_topology(
        step, &topology, None, None, fields, limits,
    )
}

pub(super) fn validate_plan(plan: &ResolvedObservationPlan) -> Result<(), ObservationError> {
    for observation in &plan.observations {
        validate_spec(&observation.spec)?;
        if !matches!(
            observation.target,
            super::ResolvedObservationTarget::Nodes { .. }
                | super::ResolvedObservationTarget::Elements { .. }
        ) {
            return Err(ObservationError::Unsupported(
                "sampled native plans require discrete native entities",
            ));
        }
    }
    Ok(())
}

pub fn evaluate_sampled_observations(
    plan: &ResolvedObservationPlan,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    fields: &RecoveredFieldCatalog,
) -> Result<Vec<ObservationResult>, ObservationError> {
    validate_plan(plan)?;
    let topology = ObservationTopology::from_sampled(sampling, mesh)?;
    super::evaluate::evaluate_observations_for_topology(plan, &topology, fields)
}

pub fn sampled_observation_partition_source_indices(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
) -> Result<Vec<usize>, ObservationError> {
    validate_plan(plan)?;
    let topology = ObservationTopology::from_sampled(sampling, mesh)?;
    super::distributed::source_selection::observation_partition_source_indices(
        plan,
        partition_plan,
        partition,
        &topology,
    )
}

pub fn prepare_sampled_observation_field_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    field: &RecoveredField,
) -> Result<RecoveredFieldPartitionSet, ObservationError> {
    validate_plan(plan)?;
    let topology = ObservationTopology::from_sampled(sampling, mesh)?;
    super::distributed::source_selection::prepare_observation_field_partition(
        plan,
        partition_plan,
        partition,
        field,
        &topology,
    )
}

pub fn evaluate_sampled_observation_partition(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    partition: &ObservationPartition,
    sampling: &SampledVolumeFieldTopology,
    mesh: &SolverMeshArtifact,
    fields: &RecoveredFieldPartitionSet,
) -> Result<ObservationPartitionContribution, ObservationError> {
    validate_plan(plan)?;
    let topology = ObservationTopology::from_sampled(sampling, mesh)?;
    if topology.digest() != plan.topology_digest {
        return Err(ObservationError::Identity(
            "sampled partition names a different sampling topology",
        ));
    }
    super::distributed::evaluate_observation_partition_with_attribution(
        plan,
        partition_plan,
        partition,
        fields,
        &topology,
        None,
    )
}

#[cfg(test)]
mod tests;
