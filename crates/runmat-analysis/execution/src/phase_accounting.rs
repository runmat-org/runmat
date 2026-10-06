//! Shared phase allocation/output accounting; no physics-family lifecycle owner.
use crate::{AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_execution_runner::AttemptResourceAccounting;

pub fn validate_analysis_phase_accounting(
    host: &AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    response: &ProgramExecutionResponse,
    execution: &crate::AnalysisPhaseExecutionEvidence,
) -> AnalysisBridgeResult<()> {
    execution.validate_for(host, plan, response)
}

pub(crate) fn validate_winning_accounting(
    host: &AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    response: &ProgramExecutionResponse,
    accounting: &AttemptResourceAccounting,
) -> AnalysisBridgeResult<()> {
    host.validate()?;
    plan.validate_identity()?;
    accounting
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = response else {
        return Err(invalid(
            "analysis phase requires externalized phase completion",
        ));
    };
    let bytes = result_objects
        .iter()
        .try_fold(0_u64, |sum, object| sum.checked_add(object.encoded_length))
        .ok_or_else(|| invalid("analysis phase output byte count overflow"))?;
    if bytes != accounting.resources.artifact_bytes
        || !runmat_analysis_fea::planning::resource::allocation_matches_plan(
            &accounting.resources.allocation,
            &plan.request,
        )
        || !host
            .workload
            .generic_capabilities()?
            .is_subset(&accounting.resources.allocation.required_capabilities)
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis phase accounting differs from its admitted request and output closure",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
