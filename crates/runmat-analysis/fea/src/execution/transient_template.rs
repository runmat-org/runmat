use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, PhysicsFamily, ProcedureStep,
    ProcedureStepId, SolutionSpace, StateKind, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{contracts::ComputeBackend, solve::transient::StructuralTimeStepPolicy};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition,
    StructuralTransientSolverPlan, StructuralTransientWorkload,
};

const TEMPLATE_PREFIX: &[u8] = b"runmat-structural-modal-transient-template/v2\0";
const TEMPLATE_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Portable modal-transient procedure before its upstream basis has been produced.
///
/// The authored intent contains this template rather than a placeholder state identity. The
/// sequence coordinator resolves the immutable state and binds its governed transition before
/// any transient task is admitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalTransientTemplate {
    pub schema_version: u16,
    pub initial_fields: Vec<super::InitialFieldBinding>,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
}

impl StructuralModalTransientTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 2
            || self.backend != ComputeBackend::Cpu
            || !self.residual_target.is_finite()
            || self.residual_target <= 0.0
            || self.time_step_policy.interval_count() == 0
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal transient template has invalid schema, backend, or solver options".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        crate::planning::resource::require_initial_field_ceiling(
            &self.initial_fields,
            TEMPLATE_LIMITS,
            step.accuracy.resources.resident_byte_budget(),
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        super::validate_initial_field_declarations(&self.model, step, &self.initial_fields)?;
        if step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != SolutionSpace::ModalReduced
            || step.family != PhysicsFamily::Structural
            || !matches!(
                step.state_inputs.as_slice(),
                [input]
                    if input.use_as == StateUse::ModalBasis
                        && input.required_kind == StateKind::ModalBasis
                        && matches!(input.source, StateSource::Upstream { .. })
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal transient template requires one structural modal-response step and one upstream modal-basis input"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matches = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matches.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "modal transient template references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "modal transient template references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    pub fn bind(
        self,
        basis_transition: CanonicalStateTransition,
    ) -> AnalysisExecutionResult<StructuralTransientWorkload> {
        self.validate()?;
        let workload = StructuralTransientWorkload {
            schema_version: self.schema_version,
            initial_fields: self.initial_fields,
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            solver: StructuralTransientSolverPlan::ModalReduced {
                basis_transition: Box::new(basis_transition),
            },
            time_step_policy: self.time_step_policy,
            residual_target: self.residual_target,
        };
        workload.validate()?;
        Ok(workload)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TEMPLATE_PREFIX,
            "structural-modal-transient-template",
            self,
            TEMPLATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let template: Self = decode_contract(
            TEMPLATE_PREFIX,
            "structural-modal-transient-template",
            bytes,
            TEMPLATE_LIMITS,
        )?;
        template.validate()?;
        Ok(template)
    }
}
