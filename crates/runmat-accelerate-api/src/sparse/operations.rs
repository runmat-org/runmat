use anyhow::{anyhow, ensure};

use crate::GpuTensorHandle;

use super::{
    ensure_vector_matches_matrix, validate_dense_owner, validate_matrix_owner,
    validate_preconditioner_owner, SparseExecutionContext, SparseFeasibilityQuery,
    SparseKrylovMethod, SparseKrylovTermination, SparseMatrixHandle, SparseOperation,
    SparseOperationEvidence, SparsePreconditionerHandle, SparsePreconditionerKind, SparseTranspose,
    SPARSE_EVIDENCE_SCHEMA_VERSION,
};

#[derive(Clone, Copy, Debug)]
pub struct SparseSpmvRequest<'a> {
    pub matrix: &'a SparseMatrixHandle,
    pub input: &'a GpuTensorHandle,
    pub transpose: SparseTranspose,
}

impl SparseSpmvRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
        validate_matrix_owner(self.matrix, provider_device_id)?;
        validate_dense_owner(self.input, provider_device_id)?;
        let input_length = match self.transpose {
            SparseTranspose::None => self.matrix.pattern.columns,
            SparseTranspose::Transpose | SparseTranspose::ConjugateTranspose => {
                self.matrix.pattern.rows
            }
        };
        ensure_vector_matches_matrix(
            self.input,
            input_length,
            self.matrix.descriptor,
            "sparse SpMV input",
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparsePreconditionerRequest<'a> {
    pub matrix: &'a SparseMatrixHandle,
    pub kind: SparsePreconditionerKind,
}

impl SparsePreconditionerRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
        validate_matrix_owner(self.matrix, provider_device_id)?;
        ensure!(
            self.matrix.pattern.rows == self.matrix.pattern.columns,
            "sparse preconditioner requires a square matrix"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparsePreconditionerApplyRequest<'a> {
    pub preconditioner: &'a SparsePreconditionerHandle,
    pub input: &'a GpuTensorHandle,
}

impl SparsePreconditionerApplyRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
        validate_preconditioner_owner(self.preconditioner, provider_device_id)?;
        validate_dense_owner(self.input, provider_device_id)?;
        ensure!(
            self.input.shape()
                == [usize::try_from(self.preconditioner.descriptor.dimension)
                    .map_err(|_| anyhow!("sparse preconditioner dimension does not fit usize"))?],
            "sparse preconditioner input shape does not match its matrix"
        );
        ensure!(
            self.input.descriptor().element_type
                == Some(self.preconditioner.descriptor.element_type)
                && self.input.descriptor().storage == Some(self.preconditioner.descriptor.storage),
            "sparse preconditioner input representation does not match its matrix"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseKrylovRequest<'a> {
    pub matrix: &'a SparseMatrixHandle,
    pub rhs: &'a GpuTensorHandle,
    pub initial_guess: Option<&'a GpuTensorHandle>,
    pub preconditioner: Option<&'a SparsePreconditionerHandle>,
    pub method: SparseKrylovMethod,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub max_iterations: u32,
    pub execution: SparseExecutionContext<'a>,
}

impl SparseKrylovRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
        self.execution.validate()?;
        validate_matrix_owner(self.matrix, provider_device_id)?;
        validate_dense_owner(self.rhs, provider_device_id)?;
        ensure_vector_matches_matrix(
            self.rhs,
            self.matrix.pattern.rows,
            self.matrix.descriptor,
            "sparse Krylov RHS",
        )?;
        if let Some(initial_guess) = self.initial_guess {
            validate_dense_owner(initial_guess, provider_device_id)?;
            ensure_vector_matches_matrix(
                initial_guess,
                self.matrix.pattern.columns,
                self.matrix.descriptor,
                "sparse Krylov initial guess",
            )?;
        }
        if let Some(preconditioner) = self.preconditioner {
            validate_preconditioner_owner(preconditioner, provider_device_id)?;
            ensure!(
                preconditioner.descriptor.matrix_id == self.matrix.matrix_id,
                "sparse preconditioner is bound to a different matrix"
            );
            ensure!(
                preconditioner.descriptor.dimension == self.matrix.pattern.rows
                    && preconditioner.descriptor.element_type
                        == self.matrix.descriptor.element_type
                    && preconditioner.descriptor.storage == self.matrix.descriptor.storage,
                "sparse preconditioner representation does not match its matrix"
            );
        }
        ensure!(
            self.matrix.pattern.rows == self.matrix.pattern.columns,
            "sparse Krylov solve requires a square matrix"
        );
        ensure!(
            self.relative_tolerance.is_finite() && self.relative_tolerance >= 0.0,
            "sparse Krylov relative tolerance must be finite and non-negative"
        );
        ensure!(
            self.absolute_tolerance.is_finite() && self.absolute_tolerance >= 0.0,
            "sparse Krylov absolute tolerance must be finite and non-negative"
        );
        ensure!(
            self.relative_tolerance > 0.0 || self.absolute_tolerance > 0.0,
            "sparse Krylov requires a positive stopping tolerance"
        );
        ensure!(
            self.max_iterations > 0,
            "sparse Krylov maximum iterations must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseKrylovResult {
    pub solution: Option<GpuTensorHandle>,
    pub termination: SparseKrylovTermination,
    pub iterations: u32,
    pub initial_residual_norm: Option<f64>,
    pub final_residual_norm: Option<f64>,
    pub evidence: SparseOperationEvidence,
}

impl SparseKrylovResult {
    pub fn validate(
        &self,
        request: SparseKrylovRequest<'_>,
        capability: &crate::ProviderCapabilitySnapshot,
    ) -> anyhow::Result<()> {
        let provider_device_id = capability.device.device_id;
        request.validate(provider_device_id)?;
        anyhow::ensure!(
            request.execution.provider_capability_schema_version == capability.schema_version,
            "sparse Krylov request capability schema version is stale"
        );
        anyhow::ensure!(
            request.execution.provider_capability_revision == capability.revision,
            "sparse Krylov request capability revision is stale"
        );
        let sparse_capability = capability
            .sparse
            .as_ref()
            .ok_or_else(|| anyhow!("provider does not advertise sparse execution"))?;
        sparse_capability.validate()?;
        anyhow::ensure!(
            sparse_capability.supports(SparseFeasibilityQuery {
                operation: SparseOperation::KrylovSolve {
                    method: request.method,
                    preconditioner: request
                        .preconditioner
                        .map(|preconditioner| preconditioner.descriptor.kind),
                },
                pattern: request.matrix.pattern,
                element_type: request.matrix.descriptor.element_type,
                storage: request.matrix.descriptor.storage,
                vector_count: 1,
                vector_length: request.matrix.pattern.rows,
                max_iterations: Some(request.max_iterations),
            }),
            "provider capability does not admit the sparse Krylov result"
        );
        let terminal_without_solution = matches!(
            self.termination,
            SparseKrylovTermination::Cancelled | SparseKrylovTermination::ProviderLost
        );
        ensure!(
            terminal_without_solution == self.solution.is_none(),
            "interrupted sparse Krylov execution must not publish a partial solution"
        );
        if let Some(solution) = &self.solution {
            validate_dense_owner(solution, provider_device_id)?;
            ensure_vector_matches_matrix(
                solution,
                request.matrix.pattern.columns,
                request.matrix.descriptor,
                "sparse Krylov solution",
            )?;
        }
        ensure!(
            self.iterations <= request.max_iterations,
            "sparse Krylov result exceeds its iteration bound"
        );
        let valid_norm =
            |value: Option<f64>| value.is_none_or(|value| value.is_finite() && value >= 0.0);
        ensure!(
            valid_norm(self.initial_residual_norm) && valid_norm(self.final_residual_norm),
            "sparse Krylov result contains an invalid residual norm"
        );
        ensure!(
            terminal_without_solution
                || (self.initial_residual_norm.is_some() && self.final_residual_norm.is_some()),
            "completed sparse Krylov result requires residual norms"
        );
        ensure!(
            !self.evidence.used_host_fallback,
            "native sparse Krylov evidence cannot report a host fallback"
        );
        ensure!(
            self.evidence.schema_version == SPARSE_EVIDENCE_SCHEMA_VERSION,
            "unsupported sparse operation evidence schema"
        );
        let expected_preconditioner = request
            .preconditioner
            .map(|preconditioner| preconditioner.preconditioner_id);
        ensure!(
            self.evidence.binding.attempt_id == request.execution.attempt_id
                && self.evidence.binding.operator_identity == request.execution.operator_identity
                && self.evidence.binding.provider_device_id == provider_device_id
                && self.evidence.binding.provider_capability_schema_version
                    == capability.schema_version
                && self.evidence.binding.provider_capability_revision == capability.revision
                && self.evidence.binding.pattern_id == request.matrix.descriptor.pattern_id
                && self.evidence.binding.matrix_id == request.matrix.matrix_id
                && self.evidence.binding.preconditioner_id == expected_preconditioner,
            "sparse operation evidence is bound to a different execution"
        );
        ensure!(
            self.evidence.peak_resident_bytes >= self.evidence.scratch_bytes,
            "sparse operation evidence understates resident storage"
        );
        let required_resident_bytes = self
            .evidence
            .resident_pattern_bytes
            .checked_add(self.evidence.resident_matrix_bytes)
            .and_then(|bytes| bytes.checked_add(self.evidence.resident_preconditioner_bytes))
            .and_then(|bytes| bytes.checked_add(self.evidence.resident_solution_bytes))
            .and_then(|bytes| bytes.checked_add(self.evidence.scratch_bytes))
            .ok_or_else(|| anyhow!("sparse operation evidence resident storage overflows"))?;
        ensure!(
            self.evidence.peak_resident_bytes >= required_resident_bytes,
            "sparse operation evidence peak is below its itemized storage"
        );
        ensure!(
            (self.termination == SparseKrylovTermination::Cancelled)
                == self.evidence.cancellation_observed,
            "sparse cancellation evidence does not match termination"
        );
        ensure!(
            (self.termination == SparseKrylovTermination::ProviderLost)
                == self.evidence.provider_loss.is_some(),
            "sparse provider-loss evidence does not match termination"
        );
        ensure!(
            (self.solution.is_none()) == (self.evidence.resident_solution_bytes == 0),
            "sparse operation evidence solution storage does not match publication"
        );
        if matches!(
            sparse_capability.cancellation,
            super::SparseCancellationCapability::Cooperative { .. }
        ) {
            ensure!(
                self.evidence.cancellation_checks > 0,
                "cooperative sparse execution must report cancellation checks"
            );
        }
        ensure!(
            sparse_capability
                .max_host_synchronizations_per_solve
                .is_some_and(|bound| self.evidence.host_synchronizations <= bound),
            "sparse operation evidence exceeds the advertised synchronization bound"
        );
        Ok(())
    }
}
