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
    // The owned numerical worker's budget; the planner chooses the solve path within it, and
    // the check refuses an estimate that cannot fit it whatever the path.
    let limit = preflight
        .estimate
        .and_then(|estimate| estimate.browser_limit(instance_budget_bytes()));
    let ready = preflight.browser_supported && publication_supported && limit.is_none();
    let refusal = limit.as_ref().map(|limit| limit.message());
    let reason = if !preflight.browser_supported {
        preflight.desktop_reason
    } else if !publication_supported {
        Some(PUBLICATION_HOST_REASON)
    } else {
        refusal.as_deref().or(Some("Runs in this browser"))
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
                memory_bytes: preflight
                    .estimate
                    .and_then(|estimate| estimate.memory_bytes),
                estimated_millis: duration.as_ref().map(|value| value.0),
                duration_evidence: duration.map(|value| value.1),
                readiness: if ready {
                    AnalysisExecutionReadiness::Ready
                } else {
                    AnalysisExecutionReadiness::DesktopRequired
                },
                reason: reason.map(str::to_owned),
                estimated_element_count: preflight.estimate.map(|estimate| estimate.element_count),
                estimated_node_count: preflight.estimate.map(|estimate| estimate.node_count),
                resource_limit: limit.map(|limit| limit.resource_limit()),
            },
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::MultiCore,
                cores: None,
                memory_bytes: None,
                estimated_millis: None,
                duration_evidence: None,
                readiness: AnalysisExecutionReadiness::DesktopRequired,
                reason: Some("Multi-core simulation runs in the desktop app.".into()),
                estimated_element_count: None,
                estimated_node_count: None,
                resource_limit: None,
            },
        ],
    })
}
