//! Borrowed authored-intent validation preserves closed original fields without cloning them.
use super::*;

pub(super) fn standalone_step(
    study: &CompiledAnalysisStudy,
) -> Result<&runmat_analysis_core::ProcedureStep, ThermalStudyPlanningError> {
    let sequence = &study.model.sequence;
    let [step] = sequence.steps.as_slice() else {
        return Err(ThermalStudyPlanningError::UnsupportedSequence);
    };
    if !sequence.dependencies.is_empty()
        || !sequence.transitions.is_empty()
        || !sequence.branches.is_empty()
    {
        return Err(ThermalStudyPlanningError::UnsupportedSequence);
    }
    Ok(step)
}

impl ThermalWorkload {
    pub fn validate_authored_standalone(
        &self,
        study: &CompiledAnalysisStudy,
        policy: ThermalStudyPlanningPolicy,
    ) -> Result<(), ThermalStudyPlanningError> {
        let step = standalone_step(study)?;
        if self.step_id != step.step_id {
            return Err(ThermalStudyPlanningError::WorkloadMismatch);
        }
        self.validate_compiled_intent(study, policy, false)
    }
    pub fn validate_authored_selected_step(
        &self,
        study: &CompiledAnalysisStudy,
        policy: ThermalStudyPlanningPolicy,
    ) -> Result<(), ThermalStudyPlanningError> {
        self.validate_compiled_intent(study, policy, true)
    }
    fn validate_compiled_intent(
        &self,
        study: &CompiledAnalysisStudy,
        policy: ThermalStudyPlanningPolicy,
        mixed_context: bool,
    ) -> Result<(), ThermalStudyPlanningError> {
        if self.model != study.model || self.backend != policy.backend {
            return Err(ThermalStudyPlanningError::WorkloadMismatch);
        }
        let (_, procedure, target) = authored_controls(&self.model, self.step_id.as_str())?;
        if self.procedure != procedure || self.residual_target != target {
            return Err(ThermalStudyPlanningError::WorkloadMismatch);
        }
        validate_sources(self, mixed_context)
    }
}
