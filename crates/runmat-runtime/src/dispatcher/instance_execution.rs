//! Select methods before reserving their asynchronous execution state.
use super::{
    finalize_instance_method_result, selection, try_call_retained_builtin_sequence_async,
    undefined_builtin_error,
};
use crate::call::BuiltinArguments;
use crate::context::RuntimeContext;
use crate::execution::{RootedValueSequence, RuntimeContinuationFuture};
use crate::RuntimeError;

/// A nonreceiver or absent method requires no continuation allocation and
/// cannot make ordinary builtin polling inherit method-execution frames.
#[inline(never)]
pub(super) fn select<'a>(
    method_name: &str,
    args: &'a BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&RuntimeContext>,
) -> Result<Option<RuntimeContinuationFuture<'a, Option<RootedValueSequence>>>, RuntimeError> {
    let selected = {
        let _entry = context.map(RuntimeContext::enter);
        selection::select_instance_method(method_name, args.view())?
    };
    let Some(selected) = selected else {
        return Ok(None);
    };
    Ok(Some(
        crate::execution::rooted_continuation::admit_rooted_future_with(
            context.cloned(),
            "instance method",
            crate::execution::rooted_continuation::semantic_control_error,
            move |context| async move { invoke(selected, args, output_count, context.as_ref()).await },
        ),
    ))
}

async fn invoke(
    selected: selection::SelectedInstanceMethod,
    args: &BuiltinArguments,
    output_count: Option<usize>,
    context: Option<&RuntimeContext>,
) -> Result<Option<RootedValueSequence>, RuntimeError> {
    let receiver = args.first().expect("instance method has a receiver");
    for route in selected.routes {
        match route {
            selection::InstanceMethodRoute::Selected(method) => {
                return crate::object::protocol::invoke_resolved_object_method_retained(
                    context,
                    &method,
                    args.fork(),
                    output_count.unwrap_or(1),
                )
                .await
                .map(Some);
            }
            selection::InstanceMethodRoute::Bound(binding) => {
                let requested = output_count.unwrap_or(1);
                let invoked = match context {
                    Some(runtime) => {
                        crate::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                            runtime, binding, args.fork().into_supplied_function_arguments(context)?, requested, None,
                        )
                        .await
                    }
                    None => {
                        crate::user_functions::try_call_rooted_bound_function(binding, args.fork().into_supplied_function_arguments(context)?, requested)
                            .await
                    }
                };
                let result = invoked.ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "UnavailableBoundFunction",
                        "registered instance method has no bound executor",
                    )
                })?;
                return finalize_instance_method_result(
                    context,
                    &selected.method,
                    receiver,
                    result,
                )
                .map(Some);
            }
            selection::InstanceMethodRoute::SemanticName(name) => {
                if let std::ops::ControlFlow::Break(result) =
                    crate::user_functions::try_call_rooted_semantic_name_in_context(
                        context,
                        &name,
                        args.fork().into_supplied_function_arguments(context)?,
                        output_count.unwrap_or(1),
                    )
                    .await
                {
                    return finalize_instance_method_result(
                        context,
                        &selected.method,
                        receiver,
                        result,
                    )
                    .map(Some);
                }
            }
            selection::InstanceMethodRoute::BuiltinName(name) => {
                let result = try_call_retained_builtin_sequence_async(
                    &name,
                    args.fork(),
                    output_count,
                    context.cloned(),
                    None,
                )
                .await
                .and_then(|result| result.ok_or_else(|| undefined_builtin_error(&name)));
                return finalize_instance_method_result(
                    context,
                    &selected.method,
                    receiver,
                    result,
                )
                .map(Some);
            }
        }
    }
    Ok(None)
}
