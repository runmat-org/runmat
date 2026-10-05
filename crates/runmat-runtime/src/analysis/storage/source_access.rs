//! Original results and authenticated metadata through the configured store owner.
use super::*;

pub fn load_run_result(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_run(run_id)
}

pub fn load_structural_frequency_result(
    run_id: &str,
) -> Result<Option<AnalysisStudyStepRunData>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_structural_frequency(run_id)
}

pub fn load_acoustic_frequency_result(
    run_id: &str,
) -> Result<Option<AnalysisStudyStepRunData>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_acoustic_frequency(run_id)
}

pub fn load_run_metadata(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_run_metadata(run_id)
}

pub fn load_assessment_metadata(
    run_id: &str,
    maximum_bytes: u64,
) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
    let guard = global_store().read().map_err(|_| {
        AssessmentStoreError::Invalid("analysis artifact store lock poisoned".into())
    })?;
    guard.load_assessment_metadata(run_id, maximum_bytes)
}
