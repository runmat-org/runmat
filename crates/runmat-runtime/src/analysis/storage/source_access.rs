//! Original results and authenticated metadata through the configured store owner.
use super::*;

pub fn load_run_result(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = capture_artifact_store()?;
    guard.load_run(&guard.resolve_run_attempt(run_id)?)
}

pub fn load_structural_frequency_result(
    run_id: &str,
) -> Result<Option<AnalysisStudyStepRunData>, String> {
    let guard = capture_artifact_store()?;
    guard.load_structural_frequency(&guard.resolve_run_attempt(run_id)?)
}

pub fn load_acoustic_frequency_result(
    run_id: &str,
) -> Result<Option<AnalysisStudyStepRunData>, String> {
    let guard = capture_artifact_store()?;
    guard.load_acoustic_frequency(&guard.resolve_run_attempt(run_id)?)
}

pub fn load_run_metadata(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = capture_artifact_store()?;
    guard.load_run_metadata(&guard.resolve_run_attempt(run_id)?)
}

pub fn load_assessment_metadata(
    run_id: &str,
    maximum_bytes: u64,
) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
    let guard = capture_artifact_store().map_err(AssessmentStoreError::Invalid)?;
    let run_id = guard
        .resolve_run_attempt(run_id)
        .map_err(AssessmentStoreError::Invalid)?;
    guard.load_assessment_metadata(&run_id, maximum_bytes)
}

/// Committed attempts of a definition run id, oldest first; the last is what the definition
/// run id resolves to.
pub fn run_attempts(definition_run_id: &str) -> Result<Vec<String>, String> {
    let guard = capture_artifact_store()?;
    guard.run_attempts(definition_run_id)
}

/// A definition run id resolves to its latest committed attempt; any other id is itself.
pub fn resolve_run_attempt(run_id: &str) -> Result<String, String> {
    let guard = capture_artifact_store()?;
    guard.resolve_run_attempt(run_id)
}
