//! Observation adapter over the existing checked-source runner and timing profile.
use super::*;

/// Timing-aware host entry. Uses the same execution path, without benchmark
/// runs, extra solves, or result publication side channels.
pub async fn execute_checked_fea_locally_with_timing(
    source: &runmat_core::CheckedFeaSessionSource,
    options: LocalFeaExecutionOptions,
    timing: &runmat_core::FeaExecutionTimingProfile,
) -> Result<LocalFeaExecution, String> {
    // The display planner describes the default singleton CPU study with one
    // worker. Never reuse sweep, accelerator, implicit-allocation, or selected
    // root measurements for that route.
    let observation = if stable_timing_source(source)
        && options.root.is_none()
        && options.workers == Some(1)
        && options.backend == runmat_analysis_fea::ComputeBackend::Cpu
        && options.accelerators.is_empty()
        && options
            .max_concurrent_cases
            .is_none_or(|count| count.get() == 1)
    {
        options
            .cpu_threads_per_worker
            .and_then(|threads| {
                timing
                    .for_source(
                        source,
                        runmat_core::FeaTimingRoute::NativeCheckedSource,
                        u32::from(threads.get()),
                    )
                    .ok()
            })
            .map(|key| key.start())
    } else {
        None
    };
    let result = execute_checked_fea_locally(source, options).await;
    if let Some(observation) = observation {
        if !runmat_runtime::interrupt::is_cancelled() {
            observation.complete(result.is_ok());
        }
    }
    result
}

// Checked text alone does not identify evaluated custom functions or generated
// geometry. Do not pool those timings as though the physical workload were fixed.
fn stable_timing_source(source: &runmat_core::CheckedFeaSessionSource) -> bool {
    let Ok(ResolvedFeaRoot::Study(study)) = source.resolve_root(None) else {
        return false;
    };
    study.study.inputs.is_empty()
        && !study.study.assessments.iter().any(|assessment| {
            matches!(
                assessment.rule,
                runmat_analysis_core::AssessmentRule::Function { .. }
            )
        })
        && matches!(
            study.geometry.source,
            runmat_geometry_core::GeometryDefinitionSource::Constructed(_)
                | runmat_geometry_core::GeometryDefinitionSource::Imported(_)
        )
}
