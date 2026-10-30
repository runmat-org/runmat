//! Result readers share the captured artifact transaction used by publication.
use super::{builtin_error_with_source, BuiltinResult, ERROR_OPERATION};

pub(super) async fn read<T>(
    builtin: &'static str,
    operation: impl FnOnce() -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    crate::analysis::storage::with_artifact_access(false, |_| operation())
        .await
        .map_err(|error| {
            builtin_error_with_source(
                builtin,
                &ERROR_OPERATION,
                "could not open simulation results",
                error,
            )
        })?
}
