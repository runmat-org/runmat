//! One readiness projection for named discovery and explicit checked-source runs.
use super::*;
use runmat_runtime::analysis::{
    AnalysisExecutionCandidate, AnalysisExecutionChoices, AnalysisExecutionMode,
};

pub(in crate::api) fn browser_fea_plan(
    session: &runmat_core::RunMatSession,
    source: &runmat_core::CheckedFeaSessionSource,
    publication_supported: bool,
) -> Result<AnalysisExecutionChoices, JsValue> {
    let preflight = source.fea_execution_preflight().map_err(|e| js_error(&e))?;
    let ready = preflight.browser_supported && publication_supported;
    // The owned numerical worker's budget; the planner chooses the solve path within it.
    let budget = instance_budget_bytes();
    let estimate = preflight
        .approximate_memory_bytes
        .filter(|bytes| *bytes > budget);
    let recommendation;
    let reason = if !preflight.browser_supported {
        preflight.desktop_reason
    } else if !publication_supported {
        Some(PUBLICATION_HOST_REASON)
    } else if let Some(bytes) = estimate {
        recommendation = format!(
            "Desktop recommended for memory: the estimate is {bytes} bytes, the browser maximum {budget} bytes."
        );
        Some(recommendation.as_str())
    } else {
        Some("Runs in this browser")
    };
    let duration = ready
        .then(|| {
            session
                .fea_execution_timing_profile()
                .for_source(source, runmat_core::FeaTimingRoute::BrowserCheckedSource, 1)
                .ok()
                .and_then(|key| key.estimate())
        })
        .flatten();
    Ok(AnalysisExecutionChoices {
        schema_version: 1,
        plan_id: uuid::Uuid::new_v4().to_string(),
        revision: source.check_result().revision.0,
        candidates: vec![
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::SingleCore,
                cores: Some(1),
                memory_bytes: preflight.approximate_memory_bytes,
                estimated_millis: duration.as_ref().map(|value| value.0),
                duration_evidence: duration.map(|value| value.1),
                readiness: if ready {
                    AnalysisExecutionReadiness::Ready
                } else {
                    AnalysisExecutionReadiness::DesktopRequired
                },
                reason: reason.map(str::to_owned),
            },
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::MultiCore,
                cores: None,
                memory_bytes: None,
                estimated_millis: None,
                duration_evidence: None,
                readiness: AnalysisExecutionReadiness::DesktopRequired,
                reason: Some("Multi-core simulation runs in the desktop app.".into()),
            },
        ],
    })
}
