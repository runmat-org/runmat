//! Select one graph-root execution before polling it. Callable, builtin and
//! parallel-region forwarding frames do not remain active around each other.
use super::*;
use futures::future::Either;
use runmat_execution::ProgramCallable;
use runmat_execution_artifact::GraphInvocationRoots;
use runmat_value::ValueSequence;
use std::future::Future;

pub(super) fn invoke_selected_root<'a>(
    request: &'a GraphExecutionRequest,
    selected: &'a SelectedWorkerGraph,
    runtime: &'a runmat_runtime::context::RuntimeContext,
    values: RootedValues,
) -> impl Future<Output = Result<RootedValueSequence, ProgramExecutionResponse>> + 'a {
    match &request.roots {
        GraphInvocationRoots::Callable { target, .. } => {
            Either::Left(callable(request, selected, runtime, values, *target))
        }
        GraphInvocationRoots::Builtin { name, .. } => Either::Right(Either::Left(builtin(
            request, selected, runtime, values, name,
        ))),
        GraphInvocationRoots::ParallelRegion { region, .. } => Either::Right(Either::Right(
            parallel_region(request, selected, runtime, values, *region),
        )),
    }
}

async fn callable(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    runtime: &runmat_runtime::context::RuntimeContext,
    values: RootedValues,
    target: GraphNodeRef,
) -> Result<RootedValueSequence, ProgramExecutionResponse> {
    let invocation = match callable_argument_layout(request.schema_version) {
        Some(CallableArgumentLayout::SuppliedOnly | CallableArgumentLayout::LegacyFlattened) => {
            // Captures and their transport schema retain the original owner.
            invoke_bound_root(request, selected, values, target).await
        }
        None => Err(runmat_runtime::runtime_error::semantic_error(
            "WorkerGraph",
            "graph callable argument schema is unsupported",
        )),
    };
    invocation.map_err(|error| graph_runtime_failure_response(error, runtime, selected))
}

async fn builtin(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    runtime: &runmat_runtime::context::RuntimeContext,
    values: RootedValues,
    name: &runmat_types::BuiltinId,
) -> Result<RootedValueSequence, ProgramExecutionResponse> {
    if !runmat_builtins::builtin_name_is_known(&name.0) {
        return Err(failure("graph builtin is absent from the worker catalog"));
    }
    let required = runmat_builtins::builtin_required_capabilities(&name.0);
    if !recipe_supports_capabilities(&request.recipe, &required) {
        return Err(failure(
            "graph recipe weakens builtin accelerator requirements",
        ));
    }
    runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
        runtime,
        &name.0,
        values.values(),
        usize::from(request.requested_outputs),
    )
    .await
    .map_err(|error| graph_runtime_failure_response(error, runtime, selected))
}

async fn parallel_region(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    runtime: &runmat_runtime::context::RuntimeContext,
    values: RootedValues,
    region: runmat_types::ParallelRegionId,
) -> Result<RootedValueSequence, ProgramExecutionResponse> {
    let Some(primary) = selected.units.first() else {
        return Err(failure("parallel graph has no primary selected unit"));
    };
    let callable = ProgramCallable::parallel_region(region);
    let value = super::super::execute_parallel_region_values(
        &callable,
        &request.context,
        request.requested_outputs,
        &request.recipe,
        primary.bytecode(),
        runtime,
        values,
    )
    .await
    .map_err(|error| match error {
        super::super::ParallelRegionError::Contract(response) => response,
        super::super::ParallelRegionError::Runtime(error) => {
            graph_runtime_failure_response(error, runtime, selected)
        }
    })?;
    value.with_owned(|value| {
        let sequence = ValueSequence::single(value)
            .map_err(|error| failure(format!("parallel graph result is invalid: {error}")))?;
        RootedValueSequence::new(sequence).map_err(|error| {
            graph_runtime_failure_response(error.into_runtime_error(), runtime, selected)
        })
    })
}
