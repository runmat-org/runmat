//! Portable canonical CPU acoustic intent. Scheduling/resource admission is a separate gate.
use super::{AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity};
use crate::{
    admission::AnalysisRunIdentity,
    contracts::{validate_solver_formulation_compatibility, SolverMethod},
    physics::acoustic::{
        solve_canonical_harmonic, validate_canonical_input, AcousticHarmonicPolicy,
    },
};
use runmat_analysis_core::{
    validate_model, AnalysisModel, AnalysisProcedure, BoundaryTransferImplementation, Frequency,
    PhysicsFamily, ProcedureStep, ProcedureStepId, SourceImplementation, StateCompleteness,
    StateKind,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};
mod fields;
pub use fields::{
    acoustic_native_field_definitions, ACOUSTIC_INTENSITY_FIELD_ID,
    ACOUSTIC_PORT_INTENSITY_FIELD_ID,
};
pub(in crate::execution) use fields::{
    native_acoustic_entity_counts, native_acoustic_field_definition, native_acoustic_field_plan,
};
mod publication;
mod result;
pub use result::{AcousticHarmonicResultSummary, AcousticHarmonicSolverResult};
mod receipt;
pub use receipt::{
    AcousticNumericalReceipt, AcousticNumericalReceiptArtifact, ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES,
};
mod sweep;
pub use sweep::{AcousticFrequencyResult, AcousticFrequencyWorkload};
mod frequency_assembly;
pub use frequency_assembly::{
    assemble_acoustic_frequency_result, AcousticFrequencyAssemblyOutput,
    AcousticFrequencyAssemblyRequest,
};
#[cfg(test)]
pub(in crate::execution) mod tests;
const PREFIX: &[u8] = b"runmat-acoustic-harmonic-workload/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticHarmonicWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub frequency: Frequency,
    pub solver: AcousticHarmonicPolicy,
    pub power_balance_relative_tolerance: f64,
}

impl AcousticHarmonicWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.step().map(|_| ())
    }
    fn step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        validate_model(&self.model).map_err(invalid)?;
        let step = validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.frequency,
            self.solver,
        )
        .map_err(invalid)?;
        if self
            .model
            .sequence
            .steps
            .iter()
            .filter(|s| s.step_id == self.step_id)
            .count()
            != 1
            || !step.activations.is_empty()
            || step.state_outputs.iter().any(|output| {
                output.kind != StateKind::AcousticComplex
                    || output.completeness != StateCompleteness::Complete
            })
        {
            return Err(invalid("canonical acoustic workload requires a unique standalone step with complete acoustic-complex outputs and without lifecycle activations"));
        }
        if !self.power_balance_relative_tolerance.is_finite()
            || self.power_balance_relative_tolerance <= 0.0
            || self.power_balance_relative_tolerance >= 1.0
        {
            return Err(invalid(
                "acoustic power tolerance must be finite and between zero and one",
            ));
        }
        Ok(step)
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-harmonic-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "acoustic-harmonic-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "acoustic-harmonic-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        self.validate()?;
        let identity = crate::admission::admit_nonstructural(
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
            Some(mesh),
            SourceImplementation::CanonicalBoundaryProjection,
            BoundaryTransferImplementation::CanonicalBoundaryProjection,
        )
        .map_err(invalid)?;
        validate_solver_formulation_compatibility(
            &identity,
            SolverMethod::AcousticCanonicalHelmholtzCertifiedLu,
            crate::contracts::SolverPreconditioner::None,
        )
        .map_err(invalid)?;
        Ok(identity)
    }
    /// Numerical execution only; this is neither a scheduler allocation nor remote/restart admission.
    /// Authored acoustic output declarations are retained in the workload identity;
    /// this entry returns physical result fields, not lifecycle state publications.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AcousticHarmonicSolverResult> {
        let formulation = self.formulation(mesh)?;
        let response = solve_canonical_harmonic(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.frequency,
            self.solver,
            self.power_balance_relative_tolerance,
        )
        .map_err(invalid)?;
        AcousticHarmonicSolverResult::new(self, mesh, formulation, response)
    }
}
fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
