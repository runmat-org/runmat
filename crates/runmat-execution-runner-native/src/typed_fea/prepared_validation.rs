use super::{LocalFeaExecutionOptions, PreparedFeaStudy};

pub(super) fn validate_prepared_execution(
    prepared: &PreparedFeaStudy,
    options: &LocalFeaExecutionOptions,
) -> Result<(), String> {
    if prepared.workers == 0 || options.workers == Some(0) {
        return Err("prepared study execution requires nonzero worker metadata".into());
    }
    // Metadata describe the original root, never a replacement scientific authority.
    let root = &prepared.root;
    if root.study != prepared.realized || root.program_revision != prepared.revision {
        return Err("prepared study metadata differs from original root study/revision".into());
    }
    if options
        .cpu_threads_per_worker
        .is_some_and(|threads| threads != root.local_execution.thread_count())
    {
        return Err("Execution threads differ from the prepared CPU admission.".into());
    }
    if options.backend != root.planning.backend()
        || root.analysis_accelerator_requests != options.accelerators
    {
        return Err("execution options differ from prepared backend/accelerator admission".into());
    }
    if crate::StudyFamilyKind::of(&root.planning).is_some()
        && (root.planning.backend() != runmat_analysis_fea::ComputeBackend::Cpu
            || !options.accelerators.is_empty())
    {
        return Err("prepared nonstructural study has no admitted accelerator envelope".into());
    }
    Ok(())
}
