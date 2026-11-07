//! Portable authored harmonic Maxwell numerical intent. Native admission is separate.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::admission::AnalysisRunIdentity;
use runmat_analysis_core::{AnalysisModel, Frequency, ProcedureStepId};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
mod admitted;
pub(in crate::execution) mod fields;
mod publication;
mod result;
pub use fields::{
    maxwell_native_field_definition_inventory, maxwell_native_field_definitions,
    MAXWELL_NATIVE_FIELD_COUNT,
};
mod receipt;
pub use receipt::{
    MaxwellNumericalReceipt, MaxwellNumericalReceiptArtifact, MAXWELL_RECEIPT_MAX_ENCODED_BYTES,
};
pub use result::MaxwellHarmonicSolverResult;
#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-maxwell-harmonic-workload/v2\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellHarmonicWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub relative_residual_target: f64,
    /// One of the step's authored frequencies; each frequency is its own workload.
    pub frequency: Frequency,
}
impl MaxwellHarmonicWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        crate::physics::maxwell::validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        let frequencies = crate::physics::maxwell::harmonic_frequencies(
            &self.model,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        if !frequencies
            .iter()
            .any(|value| value.to_bits() == self.frequency.canonical_value().to_bits())
        {
            return Err(invalid(
                "Maxwell workload frequency is not one of the step's authored frequencies",
            ));
        }
        Ok(())
    }
    /// Compile the canonical physical input at this workload's frequency.
    pub fn compile_input(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<crate::physics::maxwell::MaxwellInput> {
        crate::physics::maxwell::compile_canonical_input(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
            self.frequency.canonical_value(),
        )
        .map_err(invalid)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "maxwell-harmonic-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "maxwell-harmonic-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "maxwell-harmonic-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        self.validate()?;
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            &self.model.sequence.steps[0],
        )
    }
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        self.validate()?;
        crate::admission::admit_canonical_maxwell(mesh).map_err(invalid)
    }
    /// Serial numerical execution, not scheduler/resource or publication admission.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<MaxwellHarmonicSolverResult> {
        let input = self.compile_input(mesh)?;
        let formulation = self.formulation(mesh)?;
        let response = crate::physics::maxwell::solve(mesh, &input, self.relative_residual_target)
            .map_err(invalid)?;
        MaxwellHarmonicSolverResult::new(self, mesh, formulation, input, response)
    }
}
fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
