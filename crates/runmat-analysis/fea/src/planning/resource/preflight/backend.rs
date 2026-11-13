use crate::planning::resource::ResourcePlanningError;

/// Protocol guard for portable static task preparation. Accelerator placement is decided once by
/// `contracts::accelerator_admission` where the authored placement is resolved; a static workload
/// that still carries the GPU backend here skipped that admission.
pub fn require_executable_static_backend(
    backend: crate::ComputeBackend,
) -> Result<(), ResourcePlanningError> {
    if backend == crate::ComputeBackend::Gpu {
        return Err(ResourcePlanningError::Invalid(
            "static GPU workloads must be resolved by accelerator admission before task preparation; CPU fallback is not a GPU solve"
                .into(),
        ));
    }
    Ok(())
}
