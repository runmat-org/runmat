use crate::planning::resource::ResourcePlanningError;

/// Execution capability gate shared by portable task preparation and numerical preassembly.
/// The estimator may still model future GPU demand, but it is not an executable admission.
pub fn require_executable_static_backend(
    backend: crate::ComputeBackend,
) -> Result<(), ResourcePlanningError> {
    if backend == crate::ComputeBackend::Gpu {
        return Err(ResourcePlanningError::Invalid(
            "static GPU execution requires a native sparse solver and accelerator-bound admission; CPU fallback is not a GPU solve"
                .into(),
        ));
    }
    Ok(())
}
