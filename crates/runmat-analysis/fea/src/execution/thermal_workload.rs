//! Path-free numerical intent for one explicitly selected thermal procedure.
use runmat_analysis_core::{
    validate_model, AnalysisModel, FormulationRequirement, PhysicsFamily, ProcedureStep,
    ProcedureStepId,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{ComputeBackend, FeaRunError, FeaThermalRunResult, ThermalProcedureControl};

use super::{AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity};

const PREFIX: &[u8] = b"runmat-thermal-workload/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Placement and solver mesh are separate from authored numerical intent. This local numerical
/// entry is not a scheduler admission, a distributed solve, or a production qualification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub procedure: ThermalProcedureControl,
    pub residual_target: f64,
}

impl ThermalWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.validated_step().map(|_| ())
    }

    fn validated_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let invalid = |message: String| AnalysisExecutionError::Invalid(message);
        validate_model(&self.model).map_err(|error| invalid(error.to_string()))?;
        crate::pipeline::thermal::validate_controls(self.procedure, self.residual_target)
            .map_err(|error| invalid(error.to_string()))?;
        let mut matching = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matching
            .next()
            .filter(|_| matching.next().is_none())
            .ok_or_else(|| {
                invalid("thermal workload requires exactly one explicitly selected step".into())
            })?;
        if step.family != PhysicsFamily::Thermal || !step.activations.is_empty() {
            return Err(invalid("portable thermal workload requires standalone thermal physics without lifecycle activations".into()));
        }
        crate::pipeline::thermal::validate_step(step, self.procedure)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(step)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(PREFIX, "thermal-workload", self, LIMITS)?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(PREFIX, "thermal-workload", bytes, LIMITS)?;
        workload.validate()?;
        Ok(workload)
    }

    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "thermal-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }

    pub fn solve_schedule(&self) -> Result<super::ThermalSolveSchedule, FeaRunError> {
        let step = self
            .validated_step()
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        super::ThermalSolveSchedule::derive(&self.model, step, self.procedure)
    }

    /// Executes the selected step without rewriting the authored sequence or its producer identity.
    /// Resource planning and portable worker task admission must surround this numerical entry.
    pub fn execute(&self, mesh: &SolverMeshArtifact) -> Result<FeaThermalRunResult, FeaRunError> {
        let step = self
            .validated_step()
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        crate::pipeline::thermal::run_selected(
            &self.model,
            self.backend,
            crate::ThermalSolveOptions {
                formulation_requirement: self.formulation_requirement,
                procedure: self.procedure,
                residual_target: self.residual_target,
                solver_mesh: None,
            },
            mesh,
            step,
        )
    }
}

#[cfg(test)]
mod tests;
