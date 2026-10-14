//! The one native semantic call path for ordinary and bound nested functions.
use runmat_types::BindingId;

use crate::ExecutableUnit;

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static BOUND_NATIVE_COMPLETIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn reset_bound_native_completions() {
    BOUND_NATIVE_COMPLETIONS.with(|count| count.set(0));
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn bound_native_completions() -> usize {
    BOUND_NATIVE_COMPLETIONS.with(std::cell::Cell::get)
}

pub(super) async fn invoke(
    unit: &ExecutableUnit,
    mut request: super::super::deopt::DeoptimizationRequest,
    capture_bindings: &[BindingId],
) -> Result<runmat_runtime::execution::RootedValueSequence, runmat_runtime::RuntimeError> {
    if request.arguments.len() < capture_bindings.len() {
        return Err(super::super::error::stage(
            "NativeCaptureArity",
            "closure invocation omitted lexical captures",
        ));
    }
    request.captures = capture_bindings
        .iter()
        .copied()
        .zip(request.arguments.iter().cloned())
        .map(|(binding, value)| runmat_runtime::call::lexical::LexicalCapture { binding, value })
        .collect::<Vec<_>>();
    request.arguments = request.arguments[request.captures.len()..].to_vec();
    let requested_outputs = request.requested_outputs;
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let bound_completion_before = request
        .lexical_environment
        .as_ref()
        .map(|_| super::super::deopt::native_lexical_completions());
    let execution = super::super::deopt::invoke(unit, request).await?;
    #[cfg(all(test, not(target_arch = "wasm32")))]
    if bound_completion_before
        .is_some_and(|before| super::super::deopt::native_lexical_completions() > before)
    {
        BOUND_NATIVE_COMPLETIONS.with(|count| count.set(count.get() + 1));
    }
    execution.try_map_owned(
        |execution| super::normalize_outputs(execution.outputs, requested_outputs),
        |error| error.into_runtime_error(),
    )
}
