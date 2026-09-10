mod encoding;
mod evidence;
pub(super) mod pipelines;
mod shaders;
mod workspace;

use anyhow::{anyhow, ensure};
use runmat_accelerate_api::{
    AccelProvider, AccelProviderFuture, GpuTensorStorage, SparseExecutionBinding,
    SparseFeasibilityQuery, SparseKrylovMethod, SparseKrylovRequest, SparseKrylovResult,
    SparseKrylovTermination, SparseOperation, SparseOperationEvidence, SparseOperationTimings,
    SparsePreconditionerKind, SPARSE_EVIDENCE_SCHEMA_VERSION,
};
use runmat_time::Instant;

use self::encoding::CgInputs;
use self::workspace::{decode_state, CgTerminalState, CgWorkspace};
use super::super::{BufferUsageClass, NumericBufferRegistration, WgpuProvider};

pub(super) const ITERATIONS_PER_CHECKPOINT: u32 = 32;
pub(super) const MAXIMUM_ITERATIONS: u32 = 8_192;

pub(crate) fn execute<'a>(
    provider: &'a WgpuProvider,
    request: &'a SparseKrylovRequest<'a>,
) -> AccelProviderFuture<'a, SparseKrylovResult> {
    Box::pin(async move {
        request.validate(provider.runtime_device_id)?;
        let capability = provider.capability_snapshot();
        ensure!(
            request.execution.provider_capability_schema_version == capability.schema_version
                && request.execution.provider_capability_revision == capability.revision,
            "sparse CG request uses a stale WGPU capability"
        );
        ensure!(
            request.method == SparseKrylovMethod::ConjugateGradient
                && request
                    .preconditioner
                    .is_some_and(|value| value.descriptor.kind == SparsePreconditionerKind::Jacobi),
            "WGPU sparse CG requires its resident Jacobi preconditioner"
        );
        ensure!(
            request.max_iterations <= MAXIMUM_ITERATIONS,
            "WGPU sparse CG iteration bound exceeds its advertised envelope"
        );
        let matrix = provider.sparse.matrix(provider, request.matrix)?;
        let preconditioner = provider.sparse.preconditioner(
            provider,
            request
                .preconditioner
                .expect("checked Jacobi preconditioner"),
        )?;
        ensure!(
            preconditioner.descriptor.matrix_id == request.matrix.matrix_id,
            "WGPU sparse CG preconditioner is bound to another matrix"
        );
        let rhs = provider.get_entry(request.rhs)?;
        let initial_guess = request
            .initial_guess
            .map(|handle| provider.get_entry(handle))
            .transpose()?;
        let query = SparseFeasibilityQuery {
            operation: SparseOperation::KrylovSolve {
                method: request.method,
                preconditioner: Some(preconditioner.descriptor.kind),
            },
            pattern: matrix.pattern.descriptor,
            element_type: matrix.descriptor.element_type,
            storage: matrix.descriptor.storage,
            vector_count: 6,
            vector_length: matrix.pattern.descriptor.rows,
            max_iterations: Some(request.max_iterations),
        };
        ensure!(
            super::feasibility(provider, &query).is_supported(),
            "WGPU sparse CG exceeds the advertised provider envelope"
        );
        if let Some(reason) = provider.device_health.loss_reason() {
            return evidence::interrupted_before_dispatch(
                provider,
                request,
                &capability,
                &matrix,
                &preconditioner,
                SparseKrylovTermination::ProviderLost,
                Some(reason),
            );
        }
        if request.execution.cancellation.is_cancelled() {
            return evidence::interrupted_before_dispatch(
                provider,
                request,
                &capability,
                &matrix,
                &preconditioner,
                SparseKrylovTermination::Cancelled,
                None,
            );
        }
        let preparation_started = Instant::now();
        let dimension = usize::try_from(matrix.pattern.descriptor.rows)
            .map_err(|_| anyhow!("sparse CG dimension exceeds host address space"))?;
        let vector_groups = vector_groups(provider, dimension)?;
        let pipelines = pipelines::get(provider);
        let workspace = CgWorkspace::allocate(
            provider,
            dimension,
            vector_groups,
            request.relative_tolerance,
            request.absolute_tolerance,
            request.max_iterations,
        )?;
        let transient_reservation = provider
            .sparse
            .reserve_transient(workspace.allocated_bytes())?;
        let preparation_ns = elapsed_ns(preparation_started);

        let execution_started = Instant::now();
        let mut checkpoints = 0_u32;
        let mut cancellation_checks = 1_u64;
        let mut submitted_iterations = 0_u32;
        let mut synchronization_ns = 0_u64;
        let mut provider_loss = None;
        let mut initial_encoder = encoder(provider, "runmat-sparse-cg-initialization");
        encoding::encode_initialization(
            provider,
            &mut initial_encoder,
            &pipelines,
            CgInputs {
                matrix: &matrix,
                preconditioner: &preconditioner,
                rhs: &rhs,
                initial_guess: initial_guess.as_ref(),
            },
            &workspace,
            vector_groups,
        );
        let mut state = match submit_checkpoint(provider, initial_encoder, &workspace).await {
            Ok((snapshot, elapsed)) => {
                checkpoints += 1;
                cancellation_checks += 1;
                synchronization_ns = synchronization_ns.saturating_add(elapsed);
                snapshot
            }
            Err(failure) => {
                checkpoints += 1;
                cancellation_checks += 1;
                synchronization_ns = synchronization_ns.saturating_add(failure.elapsed_ns);
                if let Some(reason) = provider.device_health.loss_reason() {
                    provider_loss = Some(reason);
                    provider_lost_state()
                } else {
                    return Err(failure.error);
                }
            }
        };

        while state.termination.is_none() {
            if let Some(reason) = provider.device_health.loss_reason() {
                provider_loss = Some(reason);
                state.termination = Some(SparseKrylovTermination::ProviderLost);
                break;
            }
            if request.execution.cancellation.is_cancelled() {
                state.termination = Some(SparseKrylovTermination::Cancelled);
                break;
            }
            let batch = ITERATIONS_PER_CHECKPOINT
                .min(request.max_iterations.saturating_sub(submitted_iterations));
            ensure!(
                batch > 0,
                "sparse CG exhausted iterations without terminal state"
            );
            let mut encoder = encoder(provider, "runmat-sparse-cg-iteration-batch");
            for _ in 0..batch {
                encoding::encode_iteration(
                    provider,
                    &mut encoder,
                    &pipelines,
                    &matrix,
                    &preconditioner,
                    &workspace,
                    vector_groups,
                );
            }
            submitted_iterations = submitted_iterations.saturating_add(batch);
            match submit_checkpoint(provider, encoder, &workspace).await {
                Ok((snapshot, elapsed)) => {
                    checkpoints += 1;
                    cancellation_checks += 1;
                    synchronization_ns = synchronization_ns.saturating_add(elapsed);
                    state = snapshot;
                }
                Err(failure) => {
                    checkpoints += 1;
                    cancellation_checks += 1;
                    synchronization_ns = synchronization_ns.saturating_add(failure.elapsed_ns);
                    if let Some(reason) = provider.device_health.loss_reason() {
                        provider_loss = Some(reason);
                        state.termination = Some(SparseKrylovTermination::ProviderLost);
                    } else {
                        return Err(failure.error);
                    }
                }
            }
        }
        let execution_ns = elapsed_ns(execution_started);
        let termination = state
            .termination
            .ok_or_else(|| anyhow!("sparse CG completed without terminal state"))?;
        let cancelled = termination == SparseKrylovTermination::Cancelled;
        let provider_lost = termination == SparseKrylovTermination::ProviderLost;
        let interrupted = cancelled || provider_lost;
        let solution_bytes = if interrupted {
            0
        } else {
            workspace.solution.size()
        };
        let unpublished_solution_bytes = if interrupted {
            workspace.solution.size()
        } else {
            0
        };
        let checkpoint_staging_bytes = if checkpoints > 0 {
            workspace.state_size
        } else {
            0
        };
        let scratch_bytes = workspace
            .scratch_bytes()
            .saturating_add(unpublished_solution_bytes)
            .saturating_add(checkpoint_staging_bytes);
        let (pattern_bytes, matrix_bytes, preconditioner_bytes) =
            evidence::resident_input_bytes(&matrix, &preconditioner);
        let peak_resident_bytes = pattern_bytes
            .saturating_add(matrix_bytes)
            .saturating_add(preconditioner_bytes)
            .saturating_add(solution_bytes)
            .saturating_add(scratch_bytes);
        let solution = if interrupted {
            None
        } else {
            Some(provider.register_numeric_buffer(
                workspace.solution.clone(),
                NumericBufferRegistration {
                    shape: vec![dimension],
                    len: dimension,
                    physical_element_type: matrix.descriptor.element_type,
                    storage: GpuTensorStorage::Real,
                    allocated_bytes: workspace.solution.size(),
                    usage: BufferUsageClass::Generic,
                },
            ))
        };
        let result = SparseKrylovResult {
            solution,
            termination,
            iterations: state.iterations,
            initial_residual_norm: (!interrupted).then_some(state.initial_residual_norm),
            final_residual_norm: (!interrupted).then_some(state.final_residual_norm),
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
                timings: SparseOperationTimings {
                    preparation_ns,
                    execution_ns,
                    terminal_synchronization_ns: synchronization_ns,
                },
                resident_pattern_bytes: pattern_bytes,
                resident_matrix_bytes: matrix_bytes,
                resident_preconditioner_bytes: preconditioner_bytes,
                resident_solution_bytes: solution_bytes,
                host_upload_bytes: workspace.state_size,
                host_download_bytes: u64::from(checkpoints).saturating_mul(workspace.state_size),
                setup_dispatches: 1,
                spmv_dispatches: u64::from(1 + submitted_iterations),
                preconditioner_dispatches: u64::from(1 + submitted_iterations),
                vector_dispatches: u64::from(4 + submitted_iterations.saturating_mul(10)),
                host_synchronizations: checkpoints,
                device_readbacks: checkpoints,
                peak_resident_bytes,
                scratch_bytes,
                cancellation_checks,
                cancellation_observed: cancelled,
                provider_loss,
                used_host_fallback: false,
            },
        };
        result.validate(*request, &capability)?;
        drop(transient_reservation);
        Ok(result)
    })
}

async fn submit_checkpoint(
    provider: &WgpuProvider,
    mut encoder: wgpu::CommandEncoder,
    workspace: &CgWorkspace,
) -> Result<(CgTerminalState, u64), CheckpointFailure> {
    let started = Instant::now();
    let _readback_reservation = provider
        .sparse
        .reserve_transient(workspace.state_size)
        .map_err(|error| CheckpointFailure {
            error,
            elapsed_ns: elapsed_ns(started),
        })?;
    let staging =
        provider.create_readback_buffer(workspace.state_size, "runmat-sparse-cg-checkpoint");
    encoder.copy_buffer_to_buffer(
        workspace.state.as_ref(),
        0,
        staging.as_ref(),
        0,
        workspace.state_size,
    );
    provider.submit(encoder);
    let bytes = provider
        .map_readback_bytes(staging, workspace.state_size, "sparse CG checkpoint")
        .await
        .map_err(|error| CheckpointFailure {
            error,
            elapsed_ns: elapsed_ns(started),
        })?;
    let state = decode_state(provider.precision, &bytes).map_err(|error| CheckpointFailure {
        error,
        elapsed_ns: elapsed_ns(started),
    })?;
    Ok((state, elapsed_ns(started)))
}

struct CheckpointFailure {
    error: anyhow::Error,
    elapsed_ns: u64,
}

fn provider_lost_state() -> CgTerminalState {
    CgTerminalState {
        initial_residual_norm: 0.0,
        final_residual_norm: 0.0,
        iterations: 0,
        termination: Some(SparseKrylovTermination::ProviderLost),
    }
}

fn vector_groups(provider: &WgpuProvider, dimension: usize) -> anyhow::Result<u32> {
    let workgroup_size = provider.workgroup_config.scalar.max(1);
    let groups = u32::try_from(dimension)
        .ok()
        .and_then(|value| value.checked_add(workgroup_size - 1))
        .map(|value| value / workgroup_size)
        .ok_or_else(|| anyhow!("sparse CG dispatch count exceeds WGPU limits"))?;
    ensure!(
        groups > 0 && groups <= provider.adapter_limits.max_compute_workgroups_per_dimension,
        "sparse CG dispatch exceeds the WGPU workgroup limit"
    );
    Ok(groups)
}

fn encoder(provider: &WgpuProvider, label: &'static str) -> wgpu::CommandEncoder {
    provider
        .device_ref()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
