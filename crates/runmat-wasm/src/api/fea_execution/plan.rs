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
    let resident = resident_bytes();
    let preparation_fits = resident.saturating_add((512 + 128 + 64) << 20) <= BROWSER_MEMORY_BYTES;
    let ready = preflight.browser_supported && publication_supported && preparation_fits;
    let available = BROWSER_MEMORY_BYTES
        .saturating_sub(resident)
        .saturating_sub(ARTIFACT_BYTES)
        .saturating_sub(64 << 20);
    let reason = if !preflight.browser_supported {
        preflight.desktop_reason
    } else if !publication_supported {
        Some(PUBLICATION_HOST_REASON)
    } else if !preparation_fits {
        Some(
            "This session needs more memory for geometry preparation. Continue in the desktop app.",
        )
    } else if preflight
        .approximate_memory_bytes
        .is_some_and(|bytes| bytes > available)
    {
        Some("Desktop recommended for memory")
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
