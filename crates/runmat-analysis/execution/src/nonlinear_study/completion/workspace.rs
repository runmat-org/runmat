//! Original phase inventory authority and temporary cold-import lifetime.
use super::*;

fn import_workspace(
    source: &impl CacheImport,
    study: &NonlinearSolidStudyCoordinator,
    closures: &[NonlinearSolidStudyResultClosure],
) -> AnalysisBridgeResult<u64> {
    let mut phases = Vec::new();
    for closure in closures {
        let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } =
            &closure.response
        else {
            return Err(invalid(
                "original cold nonlinear phases must be externalized",
            ));
        };
        phases.push(result_objects.as_slice());
    }
    crate::study_closure::import_workspace_phases(&phases)?;
    let ceiling = study.preparation.product.workload.model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    let available = ceiling
        .checked_sub(study.admit_retention(None, None)?)
        .ok_or_else(|| limit("nonlinear import has no remaining workspace"))?;
    // Task preparation retains additional input/program state. A coarse decoder
    // bound fitting before that preparation does not establish the combined peak.
    // Always derive the current original decoder from provider-bounded artifacts.
    crate::study_closure::profile_import_workspace_phases(
        source,
        &phases,
        &study.preparation.artifact_access,
        study.preparation.output_policy.inventory_limits(),
        available,
    )
}

pub(super) fn validate(closures: &[NonlinearSolidStudyResultClosure]) -> AnalysisBridgeResult<()> {
    let phases = closures
        .iter()
        .map(|closure| match &closure.response {
            ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => {
                Ok(result_objects.as_slice())
            }
            _ => Err(invalid(
                "original cold nonlinear phases must be externalized",
            )),
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    crate::study_closure::import_workspace_phases(&phases)?;
    Ok(())
}

pub(super) fn admit(
    source: &impl CacheImport,
    study: &mut NonlinearSolidStudyCoordinator,
    closure: &NonlinearSolidStudyResultClosure,
) -> AnalysisBridgeResult<()> {
    // The preceding import has finished; its retained typed data and transport
    // are independently charged by the coordinator. No previous decode is live.
    study.import_workspace_bytes = Some(0);
    let bytes = import_workspace(source, study, std::slice::from_ref(closure))?;
    study.import_workspace_bytes = Some(bytes);
    study.admit_retention(None, None)?;
    Ok(())
}
