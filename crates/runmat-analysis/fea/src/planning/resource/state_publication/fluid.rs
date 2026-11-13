//! Complete original FV publication through the shared semantic-state phase.
use super::*;
use crate::execution::{
    fluid_state_publication_workspace_bytes, FluidSteadySolverResult, FluidSteadyWorkload,
    StatePublicationProfiles,
};
use runmat_meshing_core::SolverMeshArtifact;

/// `catalog` is the profile of exactly this request (`fluid_state_publication_profile`).
pub fn build_fluid_state_publication_resource_plan(
    request: &SemanticStatePublicationRequest,
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
    result: &FluidSteadySolverResult,
    source_payload_digest: Digest,
    catalog: &StatePublicationProfiles,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    catalog
        .validate_source(
            request.producer,
            request.source_result,
            request.topology_digest,
            source_payload_digest,
        )
        .map_err(|e| invalid(&e))?;
    let entry = catalog.entry(request).map_err(|e| invalid(&e))?;
    let bytes = result.canonical_bytes().map_err(|e| invalid(&e))?;
    let validation =
        fluid_state_publication_workspace_bytes(workload, mesh, &bytes).map_err(|e| invalid(&e))?;
    let request_identity = Digest::sha256(request.canonical_bytes().map_err(|e| invalid(&e))?);
    let mut ceiling = workload.model.sequence.steps[0].accuracy.resources;
    ceiling.set_resident_byte_budget(workload.resident_ceiling().map_err(|e| invalid(&e))?);
    let mut worker = super::super::post_process::post_process_worker();
    worker.maximum_wall_milliseconds = ceiling.maximum_wall_time_milliseconds;
    let source_workspace = validation
        .checked_add(facts.inputs.decoded_resident_bytes_admission_budget.0)
        .ok_or(ResourcePlanningError::Overflow(
            "fluid publication validation and decoded source",
        ))?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::SemanticStatePublication(request_identity),
        demand: FeaResourceDemand::SemanticStatePublication(
            SemanticStatePublicationResourceDemand {
                request: request_identity,
                source: request.source,
                source_result: request.source_result,
                source_payload_digest,
                topology_digest: request.topology_digest,
                profile_catalog: catalog.identity(),
                state_encoded_bytes_upper_bound: ByteCount(entry.encoded_bytes_upper_bound),
                state_materialization_resident_bytes_estimate: ByteCount(
                    entry.materialization_resident_bytes_estimate,
                ),
                source_validation_workspace_bytes: ByteCount(source_workspace),
                inputs: facts.inputs,
                output: facts.output,
            },
        ),
        output_set: Some(facts.output_set),
        worker,
        accuracy_ceiling: ceiling,
    })
}
