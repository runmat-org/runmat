//! Host thermal workload admission from the authoritative numerical preparation.
use super::*;
use crate::execution::{AnalysisWorkloadIdentity, ThermalWorkload};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedThermalResourceFacts {
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
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
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    if workload.backend != crate::ComputeBackend::Cpu {
        return Err(ResourcePlanningError::Invalid(
            "thermal host resource planning requires an explicit CPU backend".into(),
        ));
    }
    let numerical = workload.prepare(mesh).map_err(|e| invalid(&e))?;
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
        .checked_add(mesh.canonical_encode().map_err(|e| invalid(&e))?.len())
        .ok_or(ResourcePlanningError::Overflow("thermal input bytes"))?;
    if prepared.inputs.encoded_payload_bytes.0 != payload_bytes as u64
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
    build_resource_plan(FeaResourcePlanningInput {
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
            inputs: prepared.inputs,
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
        }),
        output_set: Some(prepared.output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: 1000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
            fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}
