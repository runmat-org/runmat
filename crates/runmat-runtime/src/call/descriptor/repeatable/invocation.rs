//! First-poll argument binding retires before the actual selected target polls.
use super::*;
use futures::future::Either;
use std::future::Future;

enum Route<'a> {
    Local(Option<&'a RuntimeContext>),
    Graph {
        runtime: &'a RuntimeContext,
        program: &'a crate::execution::DeferredGraphProgram,
        receipt: &'a crate::execution::AdmittedCallableReentry,
    },
}
struct Prepared<'a> {
    selected: CallableDescriptor<BuiltinArguments>,
    route: Route<'a>,
}
pub(super) fn run<'a>(
    owner: &'a RepeatableCallable,
    runtime: Option<&'a RuntimeContext>,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    crate::execution::after_result(
        prepare(owner, runtime, arguments, requested_outputs),
        execute,
    )
}
async fn prepare<'a>(
    owner: &'a RepeatableCallable,
    runtime: Option<&'a RuntimeContext>,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> Result<Prepared<'a>, RuntimeError> {
    let mut selected = owner.selected.clone();
    selected.requested_outputs = requested_outputs;
    let selected = selected.with_retained_arguments(arguments)?;
    let route = if let Some((program, receipt)) = &owner.reentry {
        let runtime = runtime.ok_or_else(|| {
            crate::build_runtime_error(
                "retained graph invocation requires its current explicit execution context",
            )
            .with_identifier("RunMat:Execution:SelectedCallableUnavailable")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        })?;
        Route::Graph {
            runtime,
            program,
            receipt,
        }
    } else {
        Route::Local(runtime)
    };
    Ok(Prepared { selected, route })
}
fn execute<'a>(
    prepared: Prepared<'a>,
) -> impl Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    let Prepared { selected, route } = prepared;
    match route {
        Route::Local(runtime) => Either::Left(super::super::execute_retained_callable_descriptor(
            runtime, selected,
        )),
        Route::Graph {
            runtime,
            program,
            receipt,
        } => Either::Right(
            match program.invoke_retained_reentry(runtime, receipt, selected) {
                Ok(future) => Either::Left(future),
                Err(error) => Either::Right(std::future::ready(Err(error))),
            },
        ),
    }
}
