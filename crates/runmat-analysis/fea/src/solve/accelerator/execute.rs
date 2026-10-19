use std::sync::atomic::{AtomicU64, Ordering};

use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderCancellationToken, SparseKrylovMethod, SparseKrylovRequest, SparseKrylovTermination,
};
use runmat_execution::{identity::AttemptId, Digest};

use crate::{algebra::residual, progress};

use super::{
    contract::{
        AcceleratorAdapterError, AcceleratorSolveEvidence,
        ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
    },
    prepared::AcceleratorPreparedLinearSystem,
};

static LOCAL_PROVIDER_ATTEMPT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

pub struct AcceleratorSolveOutput {
    pub solution: Vec<f64>,
    pub iterations: u32,
    pub converged: bool,
    pub evidence: AcceleratorSolveEvidence,
}

pub fn solve_prepared_accelerator_system(
    prepared: &AcceleratorPreparedLinearSystem,
    right_hand_side: &[f64],
    initial_guess: Option<&[f64]>,
    relative_tolerance: f64,
) -> Result<AcceleratorSolveOutput, AcceleratorAdapterError> {
    validate_solve_input(prepared, right_hand_side, initial_guess, relative_tolerance)?;
    let provider = prepared.provider()?;
    let shape = [prepared.prepared.dof_count()];
    let rhs = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(right_hand_side),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .map_err(|error| AcceleratorAdapterError::preparation(error.to_string()))?;
    let mut initial_values = initial_guess.map_or_else(|| vec![0.0; shape[0]], <[f64]>::to_vec);
    for (index, constrained) in prepared
        .prepared
        .constrained_dofs()
        .iter()
        .copied()
        .enumerate()
    {
        if constrained {
            initial_values[index] = right_hand_side[index];
        }
    }
    let initial = match provider.upload_numeric(&HostNumericTensorView {
        data: HostNumericDataView::F64(&initial_values),
        shape: &shape,
        storage: GpuTensorStorage::Real,
    }) {
        Ok(initial) => Some(initial),
        Err(error) => {
            return Err(AcceleratorAdapterError::preparation(error.to_string()));
        }
    };
    let cancellation = progress::cancellation_token();
    let provider_cancellation = ProviderCancellationToken::from_token(cancellation.shared_token());
    let attempt_id = provider_attempt_id(prepared.operator_identity);
    let result = (|| {
        let request = SparseKrylovRequest {
            matrix: prepared.matrix(),
            rhs: &rhs,
            initial_guess: initial.as_ref(),
            preconditioner: Some(prepared.preconditioner()),
            method: SparseKrylovMethod::ConjugateGradient,
            relative_tolerance,
            absolute_tolerance: 0.0,
            max_iterations: prepared.maximum_iterations,
            execution: runmat_accelerate_api::SparseExecutionContext {
                attempt_id,
                operator_identity: prepared.operator_identity,
                provider_capability_schema_version: prepared.capability.schema_version,
                provider_capability_revision: prepared.capability.revision,
                cancellation: &provider_cancellation,
            },
        };
        let result = futures::executor::block_on(provider.sparse_krylov_solve(&request))
            .map_err(|error| AcceleratorAdapterError::execution(error.to_string()))?;
        if let Err(error) = result.validate(request, &prepared.capability) {
            return Err(AcceleratorAdapterError::execution(error.to_string()));
        }
        Ok::<_, AcceleratorAdapterError>(result)
    })();
    // Device buffers are released by their RAII handles.
    drop(rhs);
    drop(initial);
    let result = result?;
    if result.termination == SparseKrylovTermination::Cancelled {
        let evidence = AcceleratorSolveEvidence::Cancelled {
            schema_version: ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
            operator_identity: prepared.operator_identity,
            right_hand_side_identity: numeric_identity(b"right-hand-side", right_hand_side),
            provider: result.evidence,
        };
        evidence.validate()?;
        return Err(AcceleratorAdapterError::cancellation(
            "native sparse solve observed cancellation",
            evidence,
        ));
    }
    if result.termination == SparseKrylovTermination::ProviderLost {
        let evidence = AcceleratorSolveEvidence::ProviderLost {
            schema_version: ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
            operator_identity: prepared.operator_identity,
            right_hand_side_identity: numeric_identity(b"right-hand-side", right_hand_side),
            provider: result.evidence,
        };
        evidence.validate()?;
        return Err(AcceleratorAdapterError::provider_loss(
            "native sparse provider was lost during execution",
            evidence,
        ));
    }
    let solution_handle = result.solution.as_ref().ok_or_else(|| {
        AcceleratorAdapterError::execution("provider omitted a non-cancelled solution")
    })?;
    let downloaded = futures::executor::block_on(provider.download_numeric(solution_handle))
        .map_err(|error| AcceleratorAdapterError::execution(error.to_string()));
    let downloaded = downloaded?;
    let HostNumericDataOwned::F64(solution) = downloaded.data else {
        return Err(AcceleratorAdapterError::acceptance(
            "native sparse provider changed the admitted f64 result representation",
        ));
    };
    if downloaded.storage != GpuTensorStorage::Real || downloaded.shape != shape {
        return Err(AcceleratorAdapterError::acceptance(
            "native sparse provider returned an invalid terminal solution shape",
        ));
    }
    let independent_residual = residual(prepared.prepared.matrix(), &solution, right_hand_side)
        .map_err(|error| AcceleratorAdapterError::acceptance(error.to_string()))?;
    let residual_norm = norm(&independent_residual);
    let equation_scale = norm(right_hand_side).max(1.0);
    let relative_residual = residual_norm / equation_scale;
    let provider_residual = result.final_residual_norm.ok_or_else(|| {
        AcceleratorAdapterError::acceptance(
            "non-cancelled native sparse result omitted its final residual",
        )
    })?;
    let evidence = AcceleratorSolveEvidence::Completed {
        schema_version: ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
        operator_identity: prepared.operator_identity,
        right_hand_side_identity: numeric_identity(b"right-hand-side", right_hand_side),
        solution_identity: numeric_identity(b"solution", &solution),
        provider: result.evidence,
        provider_reported_final_residual_norm: provider_residual,
        terminal_solution_readbacks: 1,
        independently_computed_residual_norm: residual_norm,
        independently_computed_relative_residual: relative_residual,
        required_relative_tolerance: relative_tolerance,
    };
    evidence.validate()?;
    let converged =
        result.termination == SparseKrylovTermination::Converged && evidence.satisfies_tolerance();
    Ok(AcceleratorSolveOutput {
        solution,
        iterations: result.iterations,
        converged,
        evidence,
    })
}

fn validate_solve_input(
    prepared: &AcceleratorPreparedLinearSystem,
    right_hand_side: &[f64],
    initial_guess: Option<&[f64]>,
    relative_tolerance: f64,
) -> Result<(), AcceleratorAdapterError> {
    if right_hand_side.len() != prepared.prepared.dof_count()
        || right_hand_side.iter().any(|value| !value.is_finite())
        || initial_guess.is_some_and(|values| {
            values.len() != prepared.prepared.dof_count()
                || values.iter().any(|value| !value.is_finite())
        })
        || !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
    {
        return Err(AcceleratorAdapterError::admission(
            "native sparse solve input is incompatible with its prepared operator",
        ));
    }
    let rebound = crate::algebra::PreparedSolverAdmissibility::bind_existing(
        prepared
            .prepared
            .admissibility()
            .ok_or_else(|| {
                AcceleratorAdapterError::admission(
                    "prepared operator has no solver admissibility evidence",
                )
            })?
            .coercivity()
            .clone(),
        prepared.prepared.matrix(),
        right_hand_side,
    )
    .map_err(|error| AcceleratorAdapterError::admission(error.to_string()))?;
    if rebound.coercivity()
        != prepared
            .prepared
            .admissibility()
            .expect("checked admissibility")
            .coercivity()
    {
        return Err(AcceleratorAdapterError::admission(
            "prepared operator coercivity evidence changed during RHS binding",
        ));
    }
    Ok(())
}

fn provider_attempt_id(operator_identity: Digest) -> AttemptId {
    let sequence = LOCAL_PROVIDER_ATTEMPT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let parent = progress::attempt_id();
    AttemptId::derive(&[
        b"fea-sparse-provider-attempt",
        parent.as_ref().map_or(&[][..], |value| value.bytes()),
        operator_identity.bytes(),
        &sequence.to_be_bytes(),
    ])
}

fn numeric_identity(domain: &[u8], values: &[f64]) -> Digest {
    let mut bytes = Vec::with_capacity(domain.len() + 8 + values.len().saturating_mul(8));
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&(values.len() as u64).to_be_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_be_bytes());
    }
    Digest::sha256(bytes)
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}
