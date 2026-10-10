//! Retire only privately owned fresh results, retaining all retirement owners.
use super::*;
pub(super) fn retire(owned: Vec<GpuTensorHandle>) -> Vec<crate::allocation::GpuHandleReleaseError> {
    owned
        .into_iter()
        .rev()
        .filter_map(|handle| handle.release().err())
        .collect()
}
pub(super) fn failure(
    primary: ProviderExecutionError,
    owned: Vec<GpuTensorHandle>,
) -> ProviderExecutionError {
    let cleanup = retire(owned);
    if cleanup.is_empty() {
        return primary;
    }
    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
        Some(primary),
        cleanup,
    ))
    .after_effect()
}
pub(super) fn failed_completion(
    output: GpuTensorHandle,
    mut cleanup: Vec<crate::allocation::GpuHandleReleaseError>,
) -> ProviderExecutionError {
    cleanup.extend(output.release().err());
    ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(None, cleanup))
        .after_effect()
}
