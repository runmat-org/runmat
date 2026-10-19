use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProvider, AccelProviderFuture, GpuTensorHandle, SparseExecutionBinding,
    SparseKrylovMethod, SparseKrylovRequest, SparseKrylovResult, SparseOperationEvidence,
    SparseOperationTimings, SparsePreconditionerApplyRequest, SparsePreconditionerHandle,
    SparsePreconditionerKind, SparsePreconditionerRequest, SparseSpmvRequest, SparseTranspose,
    SPARSE_EVIDENCE_SCHEMA_VERSION,
};
use runmat_time::Instant;

use super::{apply_jacobi, multiply, read_f64_vector, solve_cg, CgProblem};
use crate::simple_provider::InProcessProvider;

pub(in crate::simple_provider) fn spmv<'a>(
    provider: &'a InProcessProvider,
    request: &'a SparseSpmvRequest<'a>,
) -> AccelProviderFuture<'a, GpuTensorHandle> {
    Box::pin(async move {
        request.validate(provider.device_id)?;
        ensure!(
            request.transpose == SparseTranspose::None,
            "in-process sparse provider does not support transpose SpMV"
        );
        let matrix = provider.sparse.matrix(provider.device_id, request.matrix)?;
        let input = read_f64_vector(provider.device_id, request.input)?;
        let output = multiply(&matrix, &input)?;
        provider.allocate_tensor(output, vec![matrix.pattern.descriptor.rows as usize])
    })
}

pub(in crate::simple_provider) fn prepare_preconditioner<'a>(
    provider: &'a InProcessProvider,
    request: &'a SparsePreconditionerRequest<'a>,
) -> AccelProviderFuture<'a, SparsePreconditionerHandle> {
    Box::pin(async move {
        request.validate(provider.device_id)?;
        ensure!(
            request.kind == SparsePreconditionerKind::Jacobi,
            "in-process sparse provider supports Jacobi preparation only"
        );
        provider.sparse.prepare_jacobi(
            provider.device_id,
            provider.allocate_sparse_id()?,
            request.matrix,
        )
    })
}

pub(in crate::simple_provider) fn apply_preconditioner<'a>(
    provider: &'a InProcessProvider,
    request: &'a SparsePreconditionerApplyRequest<'a>,
) -> AccelProviderFuture<'a, GpuTensorHandle> {
    Box::pin(async move {
        request.validate(provider.device_id)?;
        let preconditioner = provider
            .sparse
            .preconditioner(provider.device_id, request.preconditioner)?;
        let input = read_f64_vector(provider.device_id, request.input)?;
        let output = apply_jacobi(&preconditioner, &input)?;
        provider.allocate_tensor(output, request.input.shape().to_vec())
    })
}

pub(in crate::simple_provider) fn krylov_solve<'a>(
    provider: &'a InProcessProvider,
    request: &'a SparseKrylovRequest<'a>,
) -> AccelProviderFuture<'a, SparseKrylovResult> {
    Box::pin(async move {
        request.validate(provider.device_id)?;
        let capability = provider.capability_snapshot();
        ensure!(
            request.execution.provider_capability_schema_version == capability.schema_version,
            "sparse Krylov request capability schema version is stale"
        );
        ensure!(
            request.execution.provider_capability_revision == capability.revision,
            "sparse Krylov request capability revision is stale"
        );
        ensure!(
            request.method == SparseKrylovMethod::ConjugateGradient,
            "in-process sparse provider supports conjugate gradient only"
        );
        let preparation_started = Instant::now();
        let matrix = provider.sparse.matrix(provider.device_id, request.matrix)?;
        let rhs = read_f64_vector(provider.device_id, request.rhs)?;
        let initial_guess = request
            .initial_guess
            .map(|handle| read_f64_vector(provider.device_id, handle))
            .transpose()?;
        let preconditioner = request
            .preconditioner
            .map(|handle| provider.sparse.preconditioner(provider.device_id, handle))
            .transpose()?;
        let preparation_ns = duration_ns(preparation_started);
        let execution_started = Instant::now();
        let outcome = solve_cg(CgProblem {
            matrix: &matrix,
            rhs: &rhs,
            initial_guess: initial_guess.as_deref(),
            preconditioner: preconditioner.as_deref(),
            relative_tolerance: request.relative_tolerance,
            absolute_tolerance: request.absolute_tolerance,
            max_iterations: request.max_iterations,
            cancellation: request.execution.cancellation,
        })?;
        let execution_ns = duration_ns(execution_started);
        let dimension = matrix.pattern.descriptor.rows;
        let pattern_bytes = matrix
            .pattern
            .descriptor
            .checked_storage_bytes()
            .ok_or_else(|| anyhow!("sparse pattern storage exceeds provider limits"))?;
        let value_bytes = matrix
            .descriptor
            .checked_value_bytes(matrix.pattern.descriptor.nnz)
            .ok_or_else(|| anyhow!("sparse value storage exceeds provider limits"))?;
        let preconditioner_bytes = if preconditioner.is_some() {
            dimension
                .checked_mul(8)
                .ok_or_else(|| anyhow!("sparse preconditioner storage exceeds limits"))?
        } else {
            0
        };
        let workspace_bytes = dimension
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_mul(5))
            .ok_or_else(|| anyhow!("sparse Krylov workspace exceeds provider limits"))?;
        let cancelled =
            outcome.termination == runmat_accelerate_api::SparseKrylovTermination::Cancelled;
        let resident_solution_bytes = if cancelled {
            0
        } else {
            dimension
                .checked_mul(8)
                .ok_or_else(|| anyhow!("sparse Krylov solution storage exceeds provider limits"))?
        };
        let peak_resident_bytes = pattern_bytes
            .checked_add(value_bytes)
            .and_then(|bytes| bytes.checked_add(preconditioner_bytes))
            .and_then(|bytes| bytes.checked_add(workspace_bytes))
            .and_then(|bytes| bytes.checked_add(resident_solution_bytes))
            .ok_or_else(|| anyhow!("sparse Krylov resident storage exceeds provider limits"))?;
        let solution = if cancelled {
            None
        } else {
            Some(provider.allocate_tensor(outcome.solution, vec![dimension as usize])?)
        };
        let result = SparseKrylovResult {
            solution,
            termination: outcome.termination,
            iterations: outcome.iterations,
            initial_residual_norm: outcome.initial_residual_norm,
            final_residual_norm: outcome.final_residual_norm,
            evidence: SparseOperationEvidence {
                schema_version: SPARSE_EVIDENCE_SCHEMA_VERSION,
                binding: SparseExecutionBinding {
                    attempt_id: request.execution.attempt_id,
                    operator_identity: request.execution.operator_identity,
                    provider_device_id: provider.device_id,
                    provider_capability_schema_version: capability.schema_version,
                    provider_capability_revision: capability.revision,
                    pattern_id: request.matrix.descriptor.pattern_id,
                    matrix_id: request.matrix.matrix_id,
                    preconditioner_id: request.preconditioner.map(|value| value.preconditioner_id),
                },
                timings: SparseOperationTimings {
                    preparation_ns,
                    execution_ns,
                    terminal_synchronization_ns: 0,
                },
                resident_pattern_bytes: pattern_bytes,
                resident_matrix_bytes: value_bytes,
                resident_preconditioner_bytes: preconditioner_bytes,
                resident_solution_bytes,
                host_upload_bytes: 0,
                host_download_bytes: 0,
                setup_dispatches: 0,
                spmv_dispatches: 0,
                preconditioner_dispatches: 0,
                vector_dispatches: 0,
                host_synchronizations: 0,
                device_readbacks: 0,
                peak_resident_bytes,
                scratch_bytes: workspace_bytes,
                cancellation_checks: outcome.cancellation_checks,
                cancellation_observed: cancelled,
                provider_loss: None,
                used_host_fallback: false,
            },
        };
        result.validate(*request, &capability)?;
        Ok(result)
    })
}

fn duration_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
