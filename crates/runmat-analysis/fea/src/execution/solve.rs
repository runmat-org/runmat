use runmat_analysis_core::{
    validate_model_sequence, AnalysisFieldValues, AnalysisModel, AnalysisProcedure,
    AnalysisSequence, CoupledPhysicsFamily, FormulationRequirement, PhysicsFamily, ProcedureStep,
    ProcedureStepId, RestartPolicy, SolutionSpace, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::{
    ComputeBackend, FeaElectroThermalContext, FeaRunResult, FeaThermoMechanicalContext,
};
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;

use super::CanonicalStateTransition;
use super::{AnalysisExecutionError, AnalysisExecutionResult};

const SOLVE_PREFIX: &[u8] = b"runmat-analysis-solve/v1\0";
const SOLVE_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Solver-owned, path-free input for one structural linear workload. The mesh is a separate
/// numerical object so it can be cached, transferred, and shared independently of authored state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub preconditioner: SpdPreconditionerKind,
    pub algebra_backend: LinearAlgebraBackendKind,
    pub require_solver_mesh_for_solid: bool,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    pub electro_thermal_context: Option<FeaElectroThermalContext>,
    pub initial_guess_transition: Option<CanonicalStateTransition>,
}

impl StructuralLinearWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 {
            return Err(AnalysisExecutionError::Invalid(
                "unsupported structural linear workload schema".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        if step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
            || !matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear workload requires a structural or thermomechanical static direct procedure".into(),
            ));
        }
        self.validate_state_input(step)?;
        Ok(())
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matching = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matching.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "structural linear workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matching.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural linear workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    /// Returns the single-procedure numerical projection consumed by the structural solver.
    /// The complete authored sequence remains in the portable workload and therefore in identity.
    pub fn solver_model(&self) -> AnalysisExecutionResult<AnalysisModel> {
        self.validate()?;
        let mut step = self.procedure_step()?.clone();
        step.state_inputs.clear();
        step.restart = RestartPolicy::Fresh;
        let mut model = self.model.clone();
        model.sequence = AnalysisSequence::from_steps(
            format!("{}.solver-projection", self.model.sequence.sequence_id),
            vec![step],
        );
        Ok(model)
    }

    fn validate_state_input(&self, step: &ProcedureStep) -> AnalysisExecutionResult<()> {
        if !step.activations.is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence execution does not yet implement step object activation"
                    .into(),
            ));
        }
        match &self.initial_guess_transition {
            Some(transition) => {
                transition.validate()?;
                if transition.method
                    != runmat_analysis_core::StateTransitionMethod::InitialGuessOnly
                    || transition.consumer_step_id != step.step_id
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural linear initial guess targets a different procedure step".into(),
                    ));
                }
                let matching_inputs = step
                    .state_inputs
                    .iter()
                    .filter(|input| {
                        input.use_as == StateUse::InitialGuess
                            && input.required_kind
                                == runmat_analysis_core::StateKind::StructuralPrimary
                            && matches!(
                                &input.source,
                                StateSource::Upstream {
                                    producer_step,
                                    state_id,
                                } if producer_step == &transition.producer_step_id
                                    && state_id == &transition.state_id
                            )
                    })
                    .count();
                let authored_transition = self.model.sequence.transitions.iter().any(|authored| {
                    authored.transition_id == transition.transition_id
                        && authored.producer_step == transition.producer_step_id
                        && authored.consumer_step == transition.consumer_step_id
                        && authored.state_id == transition.state_id
                        && authored.method == transition.method
                        && authored.topology == transition.topology
                        && authored.reference == transition.reference
                });
                if matching_inputs != 1 || step.state_inputs.len() != 1 || !authored_transition {
                    return Err(AnalysisExecutionError::Invalid(
                        "initial-guess transition does not exactly bind the selected step input"
                            .into(),
                    ));
                }
            }
            None if !step.state_inputs.is_empty() => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural linear step has state inputs but no executable transition".into(),
                ));
            }
            None => {}
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            bytes,
            SOLVE_LIMITS,
        )?;
        workload.validate()?;
        Ok(workload)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkloadResult {
    pub schema_version: u16,
    pub run: FeaRunResult,
}

impl StructuralLinearWorkloadResult {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.run.fields_are_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear result schema or field set is invalid".into(),
            ));
        }
        let mut field_ids = std::collections::BTreeSet::new();
        for field in &self.run.fields {
            if field.field_id.trim().is_empty() || !field_ids.insert(field.field_id.as_str()) {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural result field identities must be non-empty and unique"
                        .into(),
                ));
            }
            let AnalysisFieldValues::HostF64(values) = &field.values else {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural results must contain host-backed fields".into(),
                ));
            };
            let Some(expected_values) = field
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            else {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` shape overflows the host address space",
                    field.field_id
                )));
            };
            if values.len() != expected_values || values.iter().any(|value| !value.is_finite()) {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` has invalid shape or non-finite values",
                    field.field_id
                )));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            bytes,
            SOLVE_LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
}
