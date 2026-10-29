//! Read-only original linear fields: no operator assembly or equilibrium solve.
use super::*;
use crate::{
    constraints::ConstraintCompilation,
    execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldTopologyIdentity, StructuralLinearWorkload,
        StructuralLinearWorkloadResult,
    },
    results::field::StructuralRecoveredFieldTopology,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

pub struct StructuralFieldRecoveryPlanning<'a> {
    pub request: &'a FieldRecoveryRequest,
    pub workload: &'a StructuralLinearWorkload,
    pub source_result: &'a StructuralLinearWorkloadResult,
    pub mesh: Option<&'a SolverMeshArtifact>,
    pub compilation: &'a ConstraintCompilation<'a>,
    pub facts: PreparedPhaseResourceFacts,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: super::LocalExecutionPlan,
}
pub fn structural_field_recovery_shape(
    request: &FieldRecoveryRequest,
    result: &StructuralLinearWorkloadResult,
) -> Result<(Vec<u64>, ByteCount), ResourcePlanningError> {
    let inventory = crate::post::fields::structural_retained_field_inventory(
        &result.run.fields,
        result.constraint_result.groups.len(),
    )
    .map_err(invalid)?;
    request
        .plan
        .validate_definitions(inventory.iter().map(|(definition, _)| definition.clone()))
        .map_err(invalid)?;
    let (definitions, counts) = inventory.into_iter().unzip();
    super::recovered_field_shape::shape_with_counts(request, definitions, counts)
}

fn invalid(e: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(e.to_string())
}
pub fn build_structural_field_recovery_resource_plan(
    p: StructuralFieldRecoveryPlanning<'_>,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    p.workload.validate().map_err(invalid)?;
    p.request.validate().map_err(invalid)?;
    p.source_result
        .validate_thermal_workload(p.workload)
        .map_err(invalid)?;
    p.source_result
        .constraint_result
        .validate_against_model(p.compilation.plan(), &p.workload.model, p.mesh)
        .map_err(invalid)?;
    p.compilation
        .require_original_inputs(&p.workload.model, p.mesh)
        .map_err(invalid)?;
    let topology =
        StructuralRecoveredFieldTopology::derive(&p.workload.model, p.mesh).map_err(invalid)?;
    let constraints = p.compilation.plan();
    let step = p.workload.procedure_step().map_err(invalid)?;
    if constraints.topology != topology.identity
        || p.request.topology
            != (FieldTopologyIdentity::Structural {
                topology: topology.identity,
            })
        || p.request.plan
            != FieldRecoveryPlan::resolve_structural(step, topology, constraints.groups.len())
                .map_err(invalid)?
    {
        return Err(invalid(
            "original structural recovery topology or field plan differs",
        ));
    }
    let (partitions, construction_workspace_bytes) =
        structural_field_recovery_shape(p.request, p.source_result)?;
    let bytes = partitions.iter().try_fold(0u64, |n, b| {
        n.checked_add(*b).ok_or(ResourcePlanningError::Overflow(
            "structural recovery output",
        ))
    })?;
    if p.facts.output.encoded_payload_bytes_upper_bound.0 != bytes {
        return Err(invalid("structural recovery output shape differs"));
    }
    let mut worker = super::post_process::post_process_worker_on(p.local_execution)?;
    worker.maximum_wall_milliseconds = step.accuracy.resources.maximum_wall_time_milliseconds;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralFieldRecovery(p.request.identity()),
        demand: FeaResourceDemand::StructuralFieldRecovery(StructuralFieldRecoveryResourceDemand {
            request: p.request.identity(),
            source_result: p
                .request
                .source
                .solver_result()
                .ok_or_else(|| invalid("structural recovery requires original solver result"))?,
            topology_digest: topology.identity.discretization_digest(),
            constraint_plan_digest: Digest::sha256(constraints.canonical_bytes().map_err(invalid)?),
            constraint_compilation_peak_bytes: ByteCount(
                p.compilation.workspace().peak_requested_bytes(),
            ),
            construction_workspace_bytes,
            inputs: p.facts.inputs,
            output: p.facts.output,
        }),
        output_set: Some(p.facts.output_set),
        worker,
        accuracy_ceiling: step.accuracy.resources,
    })
}
