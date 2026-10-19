//! The one native semantic call path for ordinary and bound nested functions.
use std::rc::Rc;

use runmat_types::{BindingId, ProgramFunctionId};

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

pub(super) struct RootedInvocation {
    pub function: ProgramFunctionId,
    pub arguments: runmat_runtime::call::RootedFunctionArguments,
    pub requested_outputs: usize,
    pub runtime: runmat_runtime::context::RuntimeContext,
    pub lexical_environment: Option<runmat_value::LexicalEnvironment>,
    pub dynamic_admission: Option<runmat_vm::DynamicEvalAdmissionPort>,
}

pub(super) async fn invoke_rooted(
    unit: &ExecutableUnit,
    executor: Rc<runmat_native_executor::NativeExecutor>,
    request: RootedInvocation,
) -> Result<runmat_runtime::execution::RootedValueSequence, runmat_runtime::RuntimeError> {
    let RootedInvocation {
        function,
        arguments,
        requested_outputs,
        runtime,
        lexical_environment,
        dynamic_admission,
    } = request;
    let (runtime, placement) = runtime.native_invocation_context();
    let had_lexical_environment = lexical_environment.is_some();
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let bound_completion_before =
        had_lexical_environment.then(super::super::deopt::native_lexical_completions);
    let unit = unit.clone();
    let selected_runtime = runtime.clone();
    let values = runmat_runtime::context::run_rooted_function_native_continuation(
        &runtime,
        placement,
        arguments,
        move |arguments| {
            let continuation_runtime = selected_runtime.clone();
            let admitted = runmat_runtime::execution::admit_executable_sequence_with(
                &selected_runtime,
                move || async move {
                    let invocation = executor
                        .begin_rooted_function_arguments_async(
                            function,
                            arguments,
                            requested_outputs,
                            continuation_runtime.clone(),
                            lexical_environment,
                        )
                        .await
                        .map_err(super::super::error::from_native_executor_error)?;
                    let execution = super::super::deopt::drive_prepared_invocation(
                        &unit,
                        invocation,
                        requested_outputs,
                        continuation_runtime,
                        dynamic_admission,
                        had_lexical_environment,
                    )?
                    .await?;
                    execution.try_map_owned(
                        |execution| super::normalize_outputs(execution.outputs, requested_outputs),
                        |error| error.into_runtime_error(),
                    )
                },
            );
            async move { admitted?.await }
        },
    )
    .await?;
    #[cfg(all(test, not(target_arch = "wasm32")))]
    if bound_completion_before
        .is_some_and(|before| super::super::deopt::native_lexical_completions() > before)
    {
        BOUND_NATIVE_COMPLETIONS.with(|count| count.set(count.get() + 1));
    }
    Ok(values)
}
