//! Native choices use checked-source preflight; meshing belongs to execution.
use runmat_core::CheckedFeaSessionSource;
use runmat_runtime::analysis::{
    AnalysisExecutionCandidate, AnalysisExecutionChoices, AnalysisExecutionMode,
    AnalysisExecutionReadiness, AnalysisOperationEstimate,
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
    let estimate = preflight.estimate;
    let serial_memory = estimate.and_then(|estimate| estimate.memory_bytes);
    // A study whose estimate cannot fit the memory this machine admits a run is refused here,
    // with its numbers, instead of after meshing. Unknown capacity refuses nothing.
    let limit = estimate.and_then(|estimate| {
        let machine = crate::machine_capacity::memory_bytes().ok()?;
        estimate.native_limit(machine)
    });
    let refusal = limit.as_ref().map(|limit| {
        format!(
            "The estimated solve requires {} bytes of memory, maximum {} bytes. {}",
            limit.required_bytes.unwrap_or(limit.maximum_bytes),
            limit.maximum_bytes,
            limit.next_action
        )
    });
    // The operations the run charges, on the path this machine's memory selects. Hosts compare
    // the total with the allowance left and refuse before Run; admission refuses it too.
    let operations = estimate
        .and_then(|estimate| {
            estimate.operations(crate::machine_capacity::memory_bytes().unwrap_or(u64::MAX))
        })
        .as_ref()
        .and_then(AnalysisOperationEstimate::from_estimate);
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
                readiness: if limit.is_some() {
                    AnalysisExecutionReadiness::Unavailable
                } else {
                    AnalysisExecutionReadiness::Ready
                },
                reason: refusal.clone(),
                estimated_element_count: estimate.map(|estimate| estimate.element_count),
                estimated_node_count: estimate.map(|estimate| estimate.node_count),
                resource_limit: limit.clone(),
                estimated_operations: operations.clone(),
            },
            AnalysisExecutionCandidate {
                mode: AnalysisExecutionMode::MultiCore,
                cores: Some(cores),
                memory_bytes: parallel_memory,
                estimated_millis: parallel_duration.as_ref().map(|value| value.0),
                duration_evidence: parallel_duration.map(|value| value.1),
                readiness: if cores > 1 && limit.is_none() {
                    AnalysisExecutionReadiness::Ready
                } else {
                    AnalysisExecutionReadiness::Unavailable
                },
                reason: refusal.or_else(|| {
                    (cores < 2).then(|| "This host has one available CPU core.".into())
                }),
                estimated_element_count: estimate.map(|estimate| estimate.element_count),
                estimated_node_count: estimate.map(|estimate| estimate.node_count),
                resource_limit: limit,
                estimated_operations: operations,
            },
        ],
    })
}
