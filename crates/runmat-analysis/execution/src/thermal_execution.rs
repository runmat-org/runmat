//! Local thermal solve with output policy admitted before numerical assembly.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, PreparedAnalysisObjects};
use runmat_analysis_fea::execution::{AnalysisTransportProfile, ThermalWorkload};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ObjectNamespace};
use runmat_meshing_core::SolverMeshArtifact;

/// Admission covers conservatively retained numerical/producer/encoding components plus the
/// requested chunk-policy output closure. This is not portable worker or whole-process admission.
pub fn execute_thermal_solver_result_objects(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let prepared = workload.prepare_primary_fields(mesh)?;
    let transport = prepared.transport_profile(chunk_bytes)?;
    require_inventory(&transport, limits)?;
    let output = crate::projected_transport_output_resource_facts(&transport)?;
    prepared.require_output_reservation(output.construction_resident_bytes_admission_budget)?;
    let result = prepared.execute_primary_fields()?;
    crate::prepare_thermal_solver_result_objects(&result, chunk_bytes, namespace, limits)
}

pub(crate) fn require_inventory(
    transport: &AnalysisTransportProfile,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<()> {
    let objects = transport
        .chunk_count_upper_bound
        .checked_add(1)
        .ok_or_else(|| AnalysisBridgeError::Invalid("output object count overflow".into()))?;
    let largest = transport
        .largest_chunk_bytes_upper_bound
        .max(transport.root.encoded_bytes_upper_bound);
    limits.require_projection(
        "canonical payload projection",
        objects,
        largest,
        transport.total_bytes_upper_bound,
    )?;
    Ok(())
}
