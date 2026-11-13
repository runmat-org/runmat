//! Accepted Newton closure and original history lineage, not independent field accuracy.
//! Cold validation authenticates retained claims; numerical consumption rechecks equilibrium.
use super::*;
use crate::{
    execution::{NumericalReceiptIdentity, StepIdentity},
    solve::nonlinear::NonlinearLoadWork,
};
use runmat_canonical_codec::decode_contract;
use serde::{Deserialize, Serialize};

pub const NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES: u64 = 64 * 1024;
const PREFIX: &[u8] = b"runmat-nonlinear-solid-receipt/v3\0";
const RECEIPT_LIMITS: CanonicalLimits = CanonicalLimits::new(
    NONLINEAR_SOLID_RECEIPT_MAX_ENCODED_BYTES as usize,
    4096,
    4096,
    32,
);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidNumericalReceipt {
    pub producer: StepIdentity,
    pub topology: Digest,
    pub request: Digest,
    pub integration: Digest,
    pub physical_scope: Digest,
    pub controller: Digest,
    pub original_point_count: u64,
    pub first_output_ordinal: u64,
    pub last_output_ordinal: u64,
    pub initial_load_factor: f64,
    pub terminal_load_factor: f64,
    pub complete: bool,
    pub accepted_relative_residual_maximum: f64,
    pub accepted_relative_increment_maximum: f64,
    pub relative_increment_tolerance: f64,
    pub relative_residual_tolerance: f64,
    pub linear_relative_tolerance: f64,
    /// Largest achieved relative residual of any completed tangent solve in the original
    /// trajectory: the algebraic (inner linear) error, separate from Newton closure.
    pub maximum_linear_relative_residual: f64,
    pub accumulated_plastic_dissipation_j: f64,
    pub cumulative_work: NonlinearLoadWork,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolidNumericalReceiptArtifact {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_block: Digest,
    source_result: crate::execution::SolverResultIdentity,
    receipt: NonlinearSolidNumericalReceipt,
}

impl NonlinearSolidNumericalReceiptArtifact {
    pub fn new(
        workload: &NonlinearSolidWorkload,
        block: &NonlinearSolidBlockArtifact,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        block.validate()?;
        let source_workload = workload.identity()?;
        if block.request.workload != source_workload {
            return Err(invalid("nonlinear receipt differs from original workload"));
        }
        if block.result.solver_backend
            != crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference
            || block.result.solver_method
                != crate::contracts::SolverMethod::IncrementalNewtonRaphson
            || block.result.preconditioner != crate::contracts::SolverPreconditioner::SparseCholesky
            || !block.result.accelerator_evidence.is_empty()
            || block.result.device_apply_k_count != 0
            || block.result.device_apply_k_attempt_count != 0
        {
            return Err(invalid(
                "nonlinear receipt solver provenance differs from CPU Newton envelope",
            ));
        }
        block
            .checkpoint
            .work()
            .validate(&workload.controls)
            .map_err(invalid)?;
        let receipt = NonlinearSolidNumericalReceipt {
            producer: workload.producer_identity()?,
            topology: block.request.topology,
            request: block.request.identity()?,
            integration: block.result.accepted_state.integration_identity(),
            physical_scope: block.result.accepted_state.scope_identity(),
            controller: block.checkpoint.controller_scope(),
            original_point_count: u64::try_from(
                block.result.accepted_state.point_history().count(),
            )
            .map_err(invalid)?,
            first_output_ordinal: block.request.first_output_ordinal,
            last_output_ordinal: block.last_output_ordinal()?,
            initial_load_factor: block.result.initial_load_factor,
            terminal_load_factor: block.result.accepted_state.load_factor(),
            complete: block.complete,
            accepted_relative_residual_maximum: block.checkpoint.maximum_relative_residual(),
            relative_residual_tolerance: workload.controls.tolerance,
            accepted_relative_increment_maximum: block.checkpoint.maximum_relative_increment(),
            relative_increment_tolerance: workload.controls.tolerance,
            linear_relative_tolerance: workload.controls.linear_tolerance,
            maximum_linear_relative_residual: block.checkpoint.maximum_linear_relative_residual(),
            accumulated_plastic_dissipation_j: block
                .result
                .accepted_state
                .accumulated_plastic_dissipation_j(),
            cumulative_work: block.checkpoint.work(),
        };
        let artifact = Self {
            schema_version: 3,
            identity: NumericalReceiptIdentity::derive(
                "nonlinear-solid-numerical-receipt",
                &(source_workload, block.identity, &receipt),
            )?,
            source_workload,
            source_block: block.identity,
            source_result: block.source_result_identity()?,
            receipt,
        };
        artifact.validate()?;
        Ok(artifact)
    }
    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }
    pub fn receipt(&self) -> &NonlinearSolidNumericalReceipt {
        &self.receipt
    }
    pub fn source_block(&self) -> Digest {
        self.source_block
    }
    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.source_workload
    }
    pub const fn source_result(&self) -> crate::execution::SolverResultIdentity {
        self.source_result
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let r = &self.receipt;
        if self.schema_version != 3
            || self.source_result
                != crate::execution::SolverResultIdentity::new(
                    self.source_workload,
                    self.source_block,
                )?
            || r.original_point_count == 0
            || r.first_output_ordinal == 0
            || r.last_output_ordinal < r.first_output_ordinal
            || u64::try_from(r.cumulative_work.accepted_increments()).map_err(invalid)?
                != r.last_output_ordinal
            || !r.initial_load_factor.is_finite()
            || !r.terminal_load_factor.is_finite()
            || r.terminal_load_factor <= r.initial_load_factor
            || !r.relative_residual_tolerance.is_finite()
            || r.relative_residual_tolerance <= 0.0
            || r.relative_residual_tolerance >= 1.0
            || !r.linear_relative_tolerance.is_finite()
            || r.linear_relative_tolerance <= 0.0
            || r.linear_relative_tolerance >= 1.0
            || !r.maximum_linear_relative_residual.is_finite()
            || r.maximum_linear_relative_residual < 0.0
            || !r.accepted_relative_residual_maximum.is_finite()
            || r.accepted_relative_residual_maximum < 0.0
            || r.accepted_relative_residual_maximum > r.relative_residual_tolerance
            || !r.relative_increment_tolerance.is_finite()
            || r.relative_increment_tolerance <= 0.0
            || r.relative_increment_tolerance >= 1.0
            || !r.accepted_relative_increment_maximum.is_finite()
            || r.accepted_relative_increment_maximum < 0.0
            || r.accepted_relative_increment_maximum > r.relative_increment_tolerance
            || !r.accumulated_plastic_dissipation_j.is_finite()
            || r.accumulated_plastic_dissipation_j < 0.0
            || self.identity
                != NumericalReceiptIdentity::derive(
                    "nonlinear-solid-numerical-receipt",
                    &(self.source_workload, self.source_block, r),
                )?
        {
            return Err(invalid("invalid nonlinear accepted-closure receipt"));
        }
        Ok(())
    }
    pub fn validate_against(
        &self,
        workload: &NonlinearSolidWorkload,
        block: &NonlinearSolidBlockArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self != &Self::new(workload, block)? {
            return Err(invalid(
                "nonlinear receipt differs from original block/control/history",
            ));
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "nonlinear-solid-receipt",
            self,
            RECEIPT_LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let receipt: Self =
            decode_contract(PREFIX, "nonlinear-solid-receipt", bytes, RECEIPT_LIMITS)?;
        receipt.validate()?;
        Ok(receipt)
    }
}
