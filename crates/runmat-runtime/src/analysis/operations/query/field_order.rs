//! Result field order: the fields a result view should open first come first.
//!
//! A run lists the run fields its declared outputs read first, in output order, then every other
//! field in producer order. A study result then moves the fields its assessed outputs read ahead
//! of those, so the first descriptor is the field behind the study's verdict.
use super::*;

/// Run field ids read by the run's declared outputs, in output order, without repeats.
pub(crate) fn declared_output_source_field_ids(run_result: &AnalysisRunResult) -> Vec<String> {
    // Stored run metadata keeps the scalar inventory and drops inline observation payloads, so
    // both carry the source field id of the same outputs.
    let observed = run_result
        .run
        .observations
        .iter()
        .map(|observation| observation.source_field_id.as_ref());
    let scalars = run_result
        .observation_scalars
        .iter()
        .map(|scalar| scalar.source_field_id.as_ref());
    let mut ids: Vec<String> = Vec::new();
    for id in observed.chain(scalars).flatten() {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    ids
}

/// Stable move of the items whose field id is in `leading` to the front, in `leading` order.
pub(crate) fn move_fields_first<T>(
    items: &mut [T],
    leading: &[String],
    field_id: impl Fn(&T) -> &str,
) {
    if leading.is_empty() {
        return;
    }
    let rank = |item: &T| {
        leading
            .iter()
            .position(|id| id == field_id(item))
            .unwrap_or(leading.len())
    };
    items.sort_by_key(|item| rank(item));
}

/// Run field ids the study's assessments read for one run, in assessment order.
pub(crate) fn assessed_source_field_ids(
    assessments: &[crate::analysis::assessment::PublishedStudyAssessment],
    run_id: &str,
) -> Vec<String> {
    let mut ids = Vec::new();
    for assessment in assessments {
        if assessment.run_id.as_deref() != Some(run_id) {
            continue;
        }
        let Some(id) = assessment
            .evidence
            .as_ref()
            .and_then(|evidence| evidence.source_field_id.as_ref())
        else {
            continue;
        };
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    ids
}
