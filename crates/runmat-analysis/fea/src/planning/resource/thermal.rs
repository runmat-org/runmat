//! Host thermal workload admission from the authoritative numerical preparation.
use super::*;
use crate::execution::{AnalysisWorkloadIdentity, ThermalWorkload};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedThermalResourceFacts {
    pub local_execution: super::LocalExecutionPlan,
    pub field_source: Option<ThermalFieldSourceResourceFacts>,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

/// A host resource plan, not a claim that the thermal native task route is installed.
/// Worker execution must rederive this plan from its immutable inputs before solving.
pub fn build_thermal_resource_plan(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    chunk_bytes: usize,
    prepared: PreparedThermalResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let numerical = workload
        .prepare_primary_fields(mesh)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    build_thermal_resource_plan_from(workload, mesh, &numerical, chunk_bytes, prepared)
}

/// The same plan from a preparation the caller already derived for this workload and mesh, so a
/// host that also projects the outputs from it derives the preassembly once.
pub fn build_thermal_resource_plan_from(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    numerical: &crate::execution::PreparedThermalPrimarySolve<'_>,
    chunk_bytes: usize,
    prepared: PreparedThermalResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    if workload.backend != crate::ComputeBackend::Cpu {
        return Err(ResourcePlanningError::Invalid(
            "thermal host resource planning requires an explicit CPU backend".into(),
        ));
    }
    let transport = numerical
        .transport_profile(chunk_bytes)
        .map_err(|e| invalid(&e))?;
    let source: AnalysisWorkloadIdentity = workload.identity().map_err(|e| invalid(&e))?;
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|s| s.step_id == workload.step_id)
        .ok_or_else(|| ResourcePlanningError::Invalid("thermal step missing".into()))?;
    let payload_bytes = workload
        .canonical_bytes()
        .map_err(|e| invalid(&e))?
        .len()
        .checked_add(
            usize::try_from(mesh.canonical_encoded_len().map_err(|e| invalid(&e))?)
                .map_err(|_| ResourcePlanningError::Overflow("thermal input bytes"))?,
        )
        .ok_or(ResourcePlanningError::Overflow("thermal input bytes"))?;
    let source_bytes = prepared
        .field_source
        .as_ref()
        .map_or(0, |source| source.encoded_payload_bytes.0);
    let payload_bytes =
        (payload_bytes as u64)
            .checked_add(source_bytes)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal field-source input bytes",
            ))?;
    let profile_metadata = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))
        .map_err(|e| invalid(&e))?;
    super::post_process::validate_projected_output(
        &prepared.state_profiles_output,
        profile_metadata.encoded_bytes_upper_bound,
        "thermal state publication profiles",
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != payload_bytes
        || prepared.result_output.encoded_payload_bytes_upper_bound.0
            < transport.payload_bytes_upper_bound
        || prepared.result_output.transport_closure_bytes_upper_bound.0
            < transport.total_bytes_upper_bound
        || prepared
            .result_output
            .construction_resident_bytes_admission_budget
            .0
            < transport.total_bytes_upper_bound
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            < crate::execution::THERMAL_RECEIPT_MAX_ENCODED_BYTES
    {
        return Err(ResourcePlanningError::Invalid(
            "thermal prepared envelopes do not cover canonical inputs and outputs".into(),
        ));
    }
    let mut accuracy_ceiling = step.accuracy.resources;
    accuracy_ceiling.set_resident_byte_budget(numerical.maximum_resident_bytes());
    let plan = build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::ThermalSolve(source),
        demand: FeaResourceDemand::ThermalSolve(ThermalSolveResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            numerical_reservation_bytes: numerical.numerical_reservation_bytes()?,
            sample_count: Count(
                workload
                    .solve_schedule()
                    .map_err(|e| invalid(&e))?
                    .sample_count() as u64,
            ),
            field_source: prepared.field_source,
            inputs: prepared.inputs,
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
            state_profiles_output: prepared.state_profiles_output,
        }),
        output_set: Some(prepared.output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: prepared
                .local_execution
                .checked_cpu_millicores()
                .map_err(|error| invalid(&error))?,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
            fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling,
    })?;
    let breakdown = &plan.breakdown;
    tracing::info!(
        memory_bytes = plan.request.memory_bytes,
        input_encoded_payload_bytes = breakdown.input_encoded_payload_bytes.0,
        input_transport_closure_bytes = breakdown.input_transport_closure_bytes.0,
        input_decoded_resident_bytes = breakdown.input_decoded_resident_bytes.0,
        output_set_bytes = breakdown.output_set_bytes.0,
        workspace_bytes = breakdown.workspace_bytes.0,
        phase_resident_bytes = breakdown.phase_resident_bytes.0,
        phase_scratch_bytes = breakdown.phase_scratch_bytes.0,
        phase_retained_bytes = breakdown.phase_retained_bytes.0,
        "thermal resource plan terms"
    );
    Ok(plan)
}
