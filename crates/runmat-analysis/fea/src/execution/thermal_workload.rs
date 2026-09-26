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
pub(crate) const LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

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
        self.validate()?;
        Self::identity_for(
            &self.model,
            &self.step_id,
            self.backend,
            self.formulation_requirement,
            self.procedure,
            self.residual_target,
        )
    }

    /// Bind the exact executed intent without cloning the model. Local coupled execution
    /// shares these wire fields but does not adopt the standalone portable-workload envelope.
    pub(crate) fn identity_for(
        model: &AnalysisModel,
        step_id: &ProcedureStepId,
        backend: ComputeBackend,
        formulation_requirement: FormulationRequirement,
        procedure: ThermalProcedureControl,
        residual_target: f64,
    ) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        #[derive(Serialize)]
        struct BorrowedWorkload<'a> {
            model: &'a AnalysisModel,
            step_id: &'a ProcedureStepId,
            backend: ComputeBackend,
            formulation_requirement: FormulationRequirement,
            procedure: ThermalProcedureControl,
            residual_target: f64,
        }
        let bytes = encode_contract(
            PREFIX,
            "thermal-workload",
            &BorrowedWorkload {
                model,
                step_id,
                backend,
                formulation_requirement,
                procedure,
                residual_target,
            },
            LIMITS,
        )?;
        AnalysisWorkloadIdentity::derive("thermal-workload", &Digest::sha256(bytes))
    }

    /// Placement-independent physical model and numerical controls. Admission requirements
    /// and requested backend remain bound separately by the exact workload identity.
    pub fn numerical_intent_identity(
        &self,
    ) -> AnalysisExecutionResult<super::ThermalNumericalIntentIdentity> {
        self.validate()?;
        Self::numerical_intent_identity_for(
            &self.model,
            &self.step_id,
            self.procedure,
            self.residual_target,
        )
    }

    pub(crate) fn numerical_intent_identity_for(
        model: &AnalysisModel,
        step_id: &ProcedureStepId,
        procedure: ThermalProcedureControl,
        residual_target: f64,
    ) -> AnalysisExecutionResult<super::ThermalNumericalIntentIdentity> {
        // The shared solver view excludes display/import provenance. Its borrowed selection
        // vectors and canonical encoding are identity scratch, not allocation-free streaming.
        let bytes = encode_contract(
            PREFIX,
            "thermal-numerical-intent",
            &(
                model.solver_identity_contract(),
                step_id,
                procedure,
                residual_target,
            ),
            LIMITS,
        )?;
        super::ThermalNumericalIntentIdentity::derive(
            "thermal-numerical-intent",
            &Digest::sha256(bytes),
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
        self.prepare(mesh)?.execute()
    }

    /// Validates and reserves numerical components without assembling or solving. The returned
    /// immutable preparation allows transport admission before consuming it to execute.
    pub fn prepare<'a>(
        &'a self,
        mesh: &'a SolverMeshArtifact,
    ) -> Result<super::PreparedThermalSolve<'a>, FeaRunError> {
        let step = self
            .validated_step()
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        crate::pipeline::thermal::prepare_selected(
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
