//! Portable authored harmonic Maxwell numerical intent. Native admission is separate.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::admission::AnalysisRunIdentity;
use runmat_analysis_core::{AnalysisModel, FormulationRequirement, ProcedureStepId};
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

const PREFIX: &[u8] = b"runmat-maxwell-harmonic-workload/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellHarmonicWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub formulation_requirement: FormulationRequirement,
    pub relative_residual_target: f64,
}
impl MaxwellHarmonicWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        crate::physics::maxwell::validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.formulation_requirement,
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        Ok(())
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
        crate::admission::admit_canonical_maxwell(mesh, self.formulation_requirement)
            .map_err(invalid)
    }
    /// Serial numerical execution, not scheduler/resource or publication admission.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<MaxwellHarmonicSolverResult> {
        let input = crate::physics::maxwell::compile_canonical_input(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.formulation_requirement,
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        let formulation = self.formulation(mesh)?;
        let response = crate::physics::maxwell::solve_full_pec_volume(
            mesh,
            &input,
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        MaxwellHarmonicSolverResult::new(self, mesh, formulation, input, response)
    }
}
fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
