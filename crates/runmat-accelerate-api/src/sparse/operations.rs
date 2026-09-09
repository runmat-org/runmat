use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};

use crate::GpuTensorHandle;

use super::{
    ensure_vector_matches_matrix, validate_dense_owner, validate_matrix_owner,
    validate_preconditioner_owner, SparseKrylovMethod, SparseKrylovTermination, SparseMatrixHandle,
    SparsePreconditionerHandle, SparsePreconditionerKind, SparseTranspose,
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
            self.input.shape
                == [usize::try_from(self.preconditioner.descriptor.dimension)
                    .map_err(|_| anyhow!("sparse preconditioner dimension does not fit usize"))?],
            "sparse preconditioner input shape does not match its matrix"
        );
        ensure!(
            self.input.descriptor.element_type == Some(self.preconditioner.descriptor.element_type)
                && self.input.descriptor.storage == Some(self.preconditioner.descriptor.storage),
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
}

impl SparseKrylovRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseOperationEvidence {
    pub pattern_upload_bytes: u64,
    pub value_upload_bytes: u64,
    pub setup_dispatches: u64,
    pub spmv_dispatches: u64,
    pub preconditioner_dispatches: u64,
    pub vector_dispatches: u64,
    pub host_synchronizations: u32,
    pub device_readbacks: u32,
    pub peak_resident_bytes: u64,
    pub scratch_bytes: u64,
    pub used_host_fallback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseKrylovResult {
    pub solution: GpuTensorHandle,
    pub termination: SparseKrylovTermination,
    pub iterations: u32,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub evidence: SparseOperationEvidence,
}

impl SparseKrylovResult {
    pub fn validate(
        &self,
        request: SparseKrylovRequest<'_>,
        provider_device_id: u32,
    ) -> anyhow::Result<()> {
        request.validate(provider_device_id)?;
        validate_dense_owner(&self.solution, provider_device_id)?;
        ensure_vector_matches_matrix(
            &self.solution,
            request.matrix.pattern.columns,
            request.matrix.descriptor,
            "sparse Krylov solution",
        )?;
        ensure!(
            self.iterations <= request.max_iterations,
            "sparse Krylov result exceeds its iteration bound"
        );
        ensure!(
            self.initial_residual_norm.is_finite()
                && self.initial_residual_norm >= 0.0
                && self.final_residual_norm.is_finite()
                && self.final_residual_norm >= 0.0,
            "sparse Krylov result contains an invalid residual norm"
        );
        ensure!(
            !self.evidence.used_host_fallback,
            "native sparse Krylov evidence cannot report a host fallback"
        );
        Ok(())
    }
}
