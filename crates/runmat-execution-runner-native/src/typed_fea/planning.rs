//! Native choices use checked-source preflight; meshing belongs to execution.
use runmat_core::CheckedFeaSessionSource;
use runmat_runtime::analysis::{
    AnalysisExecutionCandidate, AnalysisExecutionChoices, AnalysisExecutionMode,
    AnalysisExecutionReadiness,
};
pub async fn plan_checked_fea_locally(
    source: &CheckedFeaSessionSource,
) -> Result<AnalysisExecutionChoices, String> {
    plan_checked_fea(source, None).await
}

pub async fn plan_checked_fea_locally_with_timing(
    source: &CheckedFeaSessionSource,
    timing: &runmat_core::FeaExecutionTimingProfile,
) -> Result<AnalysisExecutionChoices, String> {
    plan_checked_fea(source, Some(timing)).await
}

async fn plan_checked_fea(
    source: &CheckedFeaSessionSource,
    timing: Option<&runmat_core::FeaExecutionTimingProfile>,
) -> Result<AnalysisExecutionChoices, String> {
    let checked = source.check_result();
    if !checked.status.is_valid() {
        return Err("Resolve simulation checks before choosing execution options.".into());
    }
    // Choosing a host must not generate a mesh. Exact resource admission stays
    // in the existing execution preparation after the user chooses Run.
    let preflight = source.fea_execution_preflight()?;
    let serial_memory = preflight.approximate_memory_bytes;
    let cores = (crate::machine_capacity::cpu_millicores() / 1000).min(32);
    let parallel_memory = None;
    let duration = |cores| {
        timing.and_then(|profile| {
            profile
                .for_source(
                    source,
                    runmat_core::FeaTimingRoute::NativeCheckedSource,
                    cores,
                )
                .ok()?
                .estimate()
        })
    };
    let serial_duration = duration(1);
    let parallel_duration = (cores > 1).then(|| duration(cores)).flatten();
    Ok(AnalysisExecutionChoices {
        schema_version: 1,
        plan_id: runmat_execution::Digest::from_bytes(checked.revision.0).to_string(),
        revision: checked.revision.0,
        candidates: vec![
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::SingleCore,
                cores: Some(1),
                memory_bytes: serial_memory,
                estimated_millis: serial_duration.as_ref().map(|value| value.0),
                duration_evidence: serial_duration.map(|value| value.1),
                readiness: AnalysisExecutionReadiness::Ready,
                reason: None,
            },
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::MultiCore,
                cores: Some(cores),
                memory_bytes: parallel_memory,
                estimated_millis: parallel_duration.as_ref().map(|value| value.0),
                duration_evidence: parallel_duration.map(|value| value.1),
                readiness: if cores > 1 {
                    AnalysisExecutionReadiness::Ready
                } else {
                    AnalysisExecutionReadiness::Unavailable
                },
                reason: (cores < 2).then(|| "This host has one available CPU core.".into()),
            },
        ],
    })
}
