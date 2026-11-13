//! Random stream-commit terminal effect ownership.
use crate::build_runtime_error;
fn error(label: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
}
/// A committed child-stream range is a semantic effect even if the provider
/// has not submitted device work. Preserve the original language error but
/// prohibit a caller from replaying the operation on another route.
pub(crate) fn after_stream_commit(
    label: &'static str,
    source: crate::RuntimeError,
) -> crate::RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        error(label, source.message.clone()),
        runmat_accelerate_api::ProviderExecutionError::from_source(source).after_effect(),
    )
}

#[cfg(test)]
mod stream_effect_tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn committed_child_range_makes_later_failure_terminal() {
        let error = after_stream_commit("randi", error("randi", "sample allocation failed"));
        let source = error.source().expect("typed stream effect source");
        let typed = source
            .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            .expect("provider execution disposition");
        assert!(typed.is_after_effect());
        assert!(!typed.is_unsupported());
    }
}
