use runmat_accelerate_api::{
    ProviderCapabilitySnapshot, SparseExecutionBinding, SparseKrylovRequest, SparseKrylovResult,
    SparseKrylovTermination, SparseOperationEvidence, SparseOperationTimings,
    SparseProviderLossReason, SPARSE_EVIDENCE_SCHEMA_VERSION,
};

use super::super::super::WgpuProvider;
use super::super::storage::{DeviceSparseMatrix, DeviceSparsePreconditioner};

pub(super) fn resident_input_bytes(
    matrix: &DeviceSparseMatrix,
    preconditioner: &DeviceSparsePreconditioner,
) -> (u64, u64, u64) {
    let pattern = matrix
        .pattern
        .row_offsets
        .size()
        .saturating_add(matrix.pattern.column_indices.size());
    (
        pattern,
        matrix.values.size(),
        preconditioner.inverse_diagonal.size(),
    )
}

pub(super) fn interrupted_before_dispatch(
    provider: &WgpuProvider,
    request: &SparseKrylovRequest<'_>,
    capability: &ProviderCapabilitySnapshot,
    matrix: &DeviceSparseMatrix,
    preconditioner: &DeviceSparsePreconditioner,
    termination: SparseKrylovTermination,
    provider_loss: Option<SparseProviderLossReason>,
) -> anyhow::Result<SparseKrylovResult> {
    let cancelled = termination == SparseKrylovTermination::Cancelled;
    let lost = termination == SparseKrylovTermination::ProviderLost;
    anyhow::ensure!(
        cancelled || lost,
        "pre-dispatch sparse interruption requires cancellation or provider loss"
    );
    anyhow::ensure!(
        lost == provider_loss.is_some(),
        "pre-dispatch sparse provider-loss reason does not match termination"
    );
    let (pattern_bytes, matrix_bytes, preconditioner_bytes) =
        resident_input_bytes(matrix, preconditioner);
    let peak_resident_bytes = pattern_bytes
        .saturating_add(matrix_bytes)
        .saturating_add(preconditioner_bytes);
    let result = SparseKrylovResult {
        solution: None,
        termination,
        iterations: 0,
        initial_residual_norm: None,
        final_residual_norm: None,
        evidence: SparseOperationEvidence {
            schema_version: SPARSE_EVIDENCE_SCHEMA_VERSION,
            binding: SparseExecutionBinding {
                attempt_id: request.execution.attempt_id,
                operator_identity: request.execution.operator_identity,
                provider_device_id: provider.runtime_device_id,
                provider_capability_schema_version: capability.schema_version,
                provider_capability_revision: capability.revision,
                pattern_id: request.matrix.descriptor.pattern_id,
                matrix_id: request.matrix.matrix_id,
                preconditioner_id: request.preconditioner.map(|value| value.preconditioner_id),
            },
            timings: SparseOperationTimings::default(),
            resident_pattern_bytes: pattern_bytes,
            resident_matrix_bytes: matrix_bytes,
            resident_preconditioner_bytes: preconditioner_bytes,
            resident_solution_bytes: 0,
            host_upload_bytes: 0,
            host_download_bytes: 0,
            setup_dispatches: 0,
            spmv_dispatches: 0,
            preconditioner_dispatches: 0,
            vector_dispatches: 0,
            host_synchronizations: 0,
            device_readbacks: 0,
            peak_resident_bytes,
            scratch_bytes: 0,
            cancellation_checks: 1,
            cancellation_observed: cancelled,
            provider_loss,
            used_host_fallback: false,
        },
    };
    result.validate(*request, capability)?;
    Ok(result)
}
