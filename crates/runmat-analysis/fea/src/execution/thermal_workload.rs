//! Path-free numerical intent for one explicitly selected thermal procedure.
use runmat_analysis_core::{AnalysisModel, PhysicsFamily, ProcedureStep, ProcedureStepId};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::{ComputeBackend, FeaRunError, FeaThermalRunResult, ThermalProcedureControl};

use super::{AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity};

const PREFIX: &[u8] = b"runmat-thermal-workload/v1\0";
#[cfg(test)]
mod identity_tests;
mod source_identity;
mod state_profiles;
#[cfg(test)]
mod validation_tests;
/// Run data (it carries element heat sources): the platform ceiling, with the run's admitted
/// budget governing the size where the workload is produced.
pub(crate) const LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);

/// Placement and solver mesh are separate from authored numerical intent. This local numerical
/// entry is not a scheduler admission, a distributed solve, or a production qualification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub procedure: ThermalProcedureControl,
    pub residual_target: f64,
    pub initial_fields: Vec<super::InitialFieldBinding>,
}

impl ThermalWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.validated_step().map(|_| ())
    }

    pub(crate) fn validated_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        self.validated_step_with_dependency_graph_budget(u64::MAX)
    }

    pub(crate) fn validated_step_with_dependency_graph_budget(
        &self,
        maximum: u64,
    ) -> AnalysisExecutionResult<&ProcedureStep> {
        let invalid = |message: String| AnalysisExecutionError::Invalid(message);
        let maximum = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id)
            .map(|step| step.accuracy.resources.resident_byte_budget())
            .fold(maximum, u64::min);
        runmat_analysis_core::validate_model_with_dependency_graph_budget(&self.model, maximum)
            .map_err(|error| match error {
                runmat_analysis_core::AnalysisValidationError::DependencyGraphResourceLimit {
                    ..
                } => AnalysisExecutionError::Limit("thermal dependency graph workspace"),
                _ => invalid(error.to_string()),
            })?;
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
        if step.family != PhysicsFamily::Thermal
            || !crate::study::thermal_step_initial_selection_supported(step)
        {
            return Err(invalid("portable thermal workload requires thermal physics with only authored physical thermal initial selection".into()));
        }
        crate::pipeline::thermal::validate_step(step, self.procedure)
            .map_err(|error| invalid(error.to_string()))?;
        crate::planning::resource::require_initial_field_ceiling(
            &self.initial_fields,
            LIMITS,
            step.accuracy.resources.resident_byte_budget(),
        )
        .map_err(|error| invalid(error.to_string()))?;
        super::validate_initial_field_declarations(&self.model, step, &self.initial_fields)?;
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
            self.procedure,
            self.residual_target,
            &self.initial_fields,
        )
    }

    /// Bind the exact executed intent without cloning the model. Local coupled execution
    /// shares these wire fields but does not adopt the standalone portable-workload envelope.
    pub(crate) fn identity_for(
        model: &AnalysisModel,
        step_id: &ProcedureStepId,
        backend: ComputeBackend,
        procedure: ThermalProcedureControl,
        residual_target: f64,
        initial_fields: &[super::InitialFieldBinding],
    ) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        #[derive(Serialize)]
        struct BorrowedWorkload<'a> {
            model: &'a AnalysisModel,
            step_id: &'a ProcedureStepId,
            backend: ComputeBackend,
            procedure: ThermalProcedureControl,
            residual_target: f64,
            initial_fields: &'a [super::InitialFieldBinding],
        }
        let bytes = encode_contract(
            PREFIX,
            "thermal-workload",
            &BorrowedWorkload {
                model,
                step_id,
                backend,
                procedure,
                residual_target,
                initial_fields,
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
            &self.initial_fields,
        )
    }

    pub(crate) fn numerical_intent_identity_for(
        model: &AnalysisModel,
        step_id: &ProcedureStepId,
        procedure: ThermalProcedureControl,
        residual_target: f64,
        initial_fields: &[super::InitialFieldBinding],
    ) -> AnalysisExecutionResult<super::ThermalNumericalIntentIdentity> {
        // The shared solver view excludes display/import provenance. Its borrowed collection
        // views do not copy inventories; canonical encoding still requires admitted scratch.
        let bytes = encode_contract(
            PREFIX,
            "thermal-numerical-intent",
            &(
                "shared-solver-relative-residual/v2",
                model.solver_identity_contract(),
                step_id,
                procedure,
                residual_target,
                initial_fields,
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

    /// Construct the original schedule within a caller's remaining source workspace.
    /// The authored ceiling remains an additional upper bound.
    pub fn solve_schedule_admitted(
        &self,
        maximum_resident_bytes: u64,
    ) -> Result<super::ThermalSolveSchedule, FeaRunError> {
        let step = self
            .validated_step()
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        super::ThermalSolveSchedule::derive_admitted(
            &self.model,
            step,
            self.procedure,
            maximum_resident_bytes,
        )
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
        let (step, options) = self.admitted_controls(u64::MAX)?;
        crate::pipeline::thermal::prepare_selected(&self.model, self.backend, options, mesh, step)
    }
    /// Admit numerical fields and original selectors; published-field tasks admit observation execution.
    pub fn prepare_primary_fields<'a>(
        &'a self,
        mesh: &'a SolverMeshArtifact,
    ) -> Result<super::PreparedThermalPrimarySolve<'a>, FeaRunError> {
        self.prepare_primary_fields_admitted(mesh, u64::MAX)
    }

    /// Select and reserve a numerical path under a placement budget without changing authored intent.
    pub fn prepare_primary_fields_admitted<'a>(
        &'a self,
        mesh: &'a SolverMeshArtifact,
        maximum_resident_bytes: u64,
    ) -> Result<super::PreparedThermalPrimarySolve<'a>, FeaRunError> {
        let (step, options) = self.admitted_controls(maximum_resident_bytes)?;
        crate::pipeline::thermal::prepare_selected_primary(
            &self.model,
            self.backend,
            options,
            mesh,
            step,
            maximum_resident_bytes,
        )
    }
    fn admitted_controls(
        &self,
        maximum_resident_bytes: u64,
    ) -> Result<(&ProcedureStep, crate::ThermalSolveOptions), FeaRunError> {
        let step = self
            .validated_step_with_dependency_graph_budget(maximum_resident_bytes)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        crate::planning::resource::ThermalPreassemblyResourceFacts::require_input_fits(
            &self.model,
            &self.initial_fields,
            step.accuracy
                .resources
                .resident_byte_budget()
                .min(maximum_resident_bytes),
        )?;
        Ok((
            step,
            crate::ThermalSolveOptions {
                procedure: self.procedure,
                residual_target: self.residual_target,
                solver_mesh: None,
                initial_fields: self.initial_fields.clone(),
            },
        ))
    }
}

#[cfg(test)]
mod tests;
