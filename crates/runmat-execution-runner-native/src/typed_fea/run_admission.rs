//! Admission uses the same explicit worker/thread plan as native preparation.
use super::{LocalFeaExecution, LocalFeaExecutionOptions, ResolvedFeaRoot};
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
};

pub(super) struct StudyAllocation {
    pub workers: u32,
    pub threads: u32,
}

pub(super) fn study_allocation(
    study: &runmat_analysis_language::ResolvedAnalysisStudy,
    options: &LocalFeaExecutionOptions,
) -> Result<StudyAllocation, String> {
    let policy = study.execution_policy.as_ref();
    let workers = options
        .workers
        .or_else(|| {
            policy
                .and_then(|p| p.resources.workers)
                .map(|r| r.maximum.max(1))
        })
        .unwrap_or(1);
    let explicit = options
        .cpu_threads_per_worker
        .map(|v| u32::from(v.get()))
        .or_else(|| policy.and_then(|p| p.resources.cpu_threads_per_worker));
    let threads = crate::machine_capacity::study_allocation(workers, explicit)?;
    Ok(StudyAllocation { workers, threads })
}

pub(super) fn admit_root(
    source: &runmat_core::CheckedFeaSessionSource,
    root: &ResolvedFeaRoot,
    mut options: LocalFeaExecutionOptions,
) -> Result<(FeaRunScope, LocalFeaExecutionOptions), String> {
    let (study, root_id, cases) = match root {
        ResolvedFeaRoot::Study(study) => (
            study,
            study.study.study_id.as_str(),
            super::harmonic_frequency_cases(study).len().max(1),
        ),
        ResolvedFeaRoot::Sweep(sweep) => (
            &sweep.study,
            sweep.sweep.sweep_id.as_str(),
            sweep
                .sweep
                .expand_cases(&sweep.study.study)
                .map_err(|e| e.to_string())?
                .len(),
        ),
    };
    let allocation = study_allocation(study, &options)?;
    // Freeze the same plan into execution so admission and preparation cannot
    // resolve different defaults, even if host capacity changes between them.
    options.workers = Some(allocation.workers);
    options.cpu_threads_per_worker = Some(
        runmat_analysis_fea::planning::resource::LocalThreadCount::new(
            u16::try_from(allocation.threads)
                .map_err(|_| "FEA CPU allocation exceeds local thread bound")?,
        )
        .map_err(|e| e.to_string())?,
    );
    let concurrency = match root {
        ResolvedFeaRoot::Sweep(sweep) => super::sweep_concurrency(sweep, &options, cases),
        ResolvedFeaRoot::Study(_) => crate::machine_capacity::case_concurrency(
            allocation.workers,
            allocation.threads,
            cases,
            options.max_concurrent_cases.map(|v| usize::from(v.get())),
        ),
    };
    let millicores = u64::from(allocation.workers)
        .checked_mul(u64::from(allocation.threads))
        .and_then(|v| v.checked_mul(concurrency as u64))
        .and_then(|v| v.checked_mul(1000))
        .ok_or("FEA CPU allocation overflow")?;
    if runmat_runtime::analysis::fea_run_admission::current_fea_cpu_capacity_millicores()
        .is_some_and(|ceiling| millicores > ceiling)
    {
        return Err(
            "FEA worker/thread/case allocation exceeds the original admitted CPU capacity".into(),
        );
    }
    let mut identity = source.check_result().revision.0.to_vec();
    identity.extend_from_slice(root_id.as_bytes());
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        millicores,
        runmat_execution::Digest::sha256(identity),
    )
    .map_err(|e| format!("{}: {e}", e.error_code()))?;
    request.site = FeaRunScope::admission_site().map_err(|e| format!("{}: {e}", e.error_code()))?;
    let scope = FeaRunScope::admit(request).map_err(|e| format!("{}: {e}", e.error_code()))?;
    Ok((scope, options))
}

pub(super) fn outcome(result: &Result<LocalFeaExecution, String>) -> FeaRunOutcome {
    if runmat_runtime::interrupt::is_cancelled() {
        return FeaRunOutcome::Cancelled;
    }
    match result {
        Ok(value) if value.is_successful() => FeaRunOutcome::Completed,
        Ok(_) | Err(_) => FeaRunOutcome::Failed,
    }
}
