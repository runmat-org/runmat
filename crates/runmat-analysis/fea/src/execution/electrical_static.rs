//! Portable static scalar FEM contracts. Numerical execution only: no scheduler allocation,
//! Native retained-field/state contracts do not imply restart or independent accuracy promotion.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::admission::AnalysisRunIdentity;
use runmat_analysis_core::{
    AnalysisModel, ElectromagneticDomain, ProcedureStepId, StaticElectricalEquation,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

mod admitted;
mod result;
pub use result::ElectricalStaticSolverResult;
pub(in crate::execution) mod fields;
pub(in crate::execution) use fields::{
    native_electrical_entity_counts, native_electrical_field_definition,
    native_electrical_field_plan,
};
mod publication;
mod receipt;
pub use receipt::{
    ElectricalNumericalReceipt, ElectricalNumericalReceiptArtifact,
    ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES,
};
#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-electrical-static-workload/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStaticWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub relative_residual_target: f64,
}

impl ElectricalStaticWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        crate::physics::electrical::validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        Ok(())
    }
    pub fn equation(&self) -> AnalysisExecutionResult<StaticElectricalEquation> {
        self.validate()?;
        match &self.model.electromagnetic {
            Some(ElectromagneticDomain::Static { equation }) => Ok(*equation),
            _ => Err(invalid("static electrical equation required")),
        }
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "electrical-static-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "electrical-static-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "electrical-static-workload",
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
        crate::admission::admit_canonical_electrical_static(
            self.equation()?,
            crate::physics::electrical::sources::kind(&self.model),
            mesh,
        )
        .map_err(invalid)
    }
    /// Executes the existing canonical CPU numerical kernel. This is not an admitted
    /// resource plan or semantic study completion. Result import never reruns this solve.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<ElectricalStaticSolverResult> {
        let formulation = self.formulation(mesh)?;
        let response = crate::physics::electrical::solve_canonical_static(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        ElectricalStaticSolverResult::new(self, mesh, formulation, response)
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
