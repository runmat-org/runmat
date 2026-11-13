//! Callable forwarding retains original argument owners and typed outputs.
use crate::call::BuiltinArguments;
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueSequence, RootedValues};
use crate::{runtime_descriptor_error_with_detail, RuntimeError};
use runmat_builtins::FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED;
use runmat_value::{Value, ValueSequence};

#[cfg(test)]
#[path = "feval/roots_tests.rs"]
mod roots_tests;

#[cfg(test)]
#[path = "feval/named_tests.rs"]
mod named_tests;

pub(crate) fn admit_inputs(
    target: &Value,
    arguments: &[Value],
) -> Result<
    (
        crate::execution::RootedValueLease,
        crate::execution::RootedValueLease,
    ),
    crate::RuntimeError,
> {
    let target_root = crate::execution::RootedValueLease::for_traced(target)
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    let argument_roots = crate::execution::RootedValueLease::for_values(arguments.iter())
        .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    Ok((target_root, argument_roots))
}

/// Compatibility for an existing owned ordinary embedding batch.
pub(crate) async fn invoke(
    runtime: Option<&RuntimeContext>,
    invocation_outputs: Option<usize>,
    target: Value,
    arguments: Vec<Value>,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> crate::BuiltinResult<RootedValueSequence> {
    let arguments = match runtime {
        Some(runtime) => RootedValues::in_context(arguments, runtime)?,
        None => RootedValues::new(arguments).map_err(|e| e.into_runtime_error())?,
    };
    invoke_retained(
        runtime,
        target,
        arguments.into(),
        invocation_outputs
            .or_else(crate::output_count::current_output_count)
            .unwrap_or(1),
        input_callsite,
    )
    .await
}

/// Acquire the target's source registration at future creation, before an
/// unpolled callback can outlive the frame that supplied its target.
pub(crate) fn invoke_retained<'a>(
    runtime: Option<&'a RuntimeContext>,
    target: Value,
    arguments: BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = crate::BuiltinResult<RootedValueSequence>> + 'a {
    let target = match runtime {
        Some(runtime) => RootedValue::in_context(target, runtime),
        None => {
            use crate::context::host_allocation::{
                host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
                HostAllocationReceipt,
            };
            let error = |source| {
                host_allocation_runtime_error(
                    None,
                    "RunMat:Call:Arguments",
                    "callback target policy unavailable",
                    source,
                )
            };
            HostAllocationReceipt::policy(None)
                .map_err(error)
                .and_then(|policy| {
                    if policy != HostAllocationPolicy::Unbounded {
                        return Err(error(HostAllocationError::MissingManagedOwner));
                    }
                    RootedValue::new(target).map_err(|e| e.into_runtime_error())
                })
        }
    };
    async move {
        invoke_rooted_target(
            runtime,
            target?,
            arguments,
            requested_outputs,
            input_callsite,
        )
        .await
    }
}

/// Same entry for a target whose existing registration must not be retraced.
pub(crate) fn invoke_rooted_target<'a>(
    runtime: Option<&'a RuntimeContext>,
    target: RootedValue,
    arguments: BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = crate::BuiltinResult<RootedValueSequence>> + 'a {
    invoke_selected(
        runtime,
        Ok(target),
        arguments,
        requested_outputs,
        input_callsite,
    )
}

fn invoke_selected<'a>(
    runtime: Option<&'a RuntimeContext>,
    target: Result<RootedValue, RuntimeError>,
    arguments: BuiltinArguments,
    requested_outputs: usize,
    input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> impl std::future::Future<Output = crate::BuiltinResult<RootedValueSequence>> + 'a {
    let selected = runmat_value::validate_output_count(requested_outputs)
        .map_err(crate::sequence::sequence_error_to_runtime)
        .and(target)
        .and_then(|target| {
            crate::call::descriptor::RepeatableCallable::select_deferred_rooted(runtime, target)
        })
        .map(|selected| selected.with_callsite(input_callsite));
    async move {
        let selected = selected?;
        let invoke = async {
            let _guard = crate::output_count::push_output_count(Some(requested_outputs))?;
            Box::pin(selected.invoke_retained(runtime, arguments, requested_outputs)).await
        };
        match runtime {
            Some(runtime) => runtime.scope(invoke).await,
            None => invoke.await,
        }
    }
}

/// Refine character/scalar-string syntax into one canonical immutable text
/// owner, preserving feval's existing rows/element-count acceptance rules.
pub(crate) fn refine_target(
    target: RootedValue,
    runtime: Option<&RuntimeContext>,
) -> Result<RootedValue, RuntimeError> {
    crate::call::admission::admit_feval_target(target.value())?;
    let text = match target.value() {
        Value::CharArray(array) if array.rows == 1 => Some(crate::call::argument_text::copy_chars(
            array.data(),
            runtime,
        )?),
        Value::StringArray(array) if array.data().len() == 1 => Some(
            crate::call::argument_text::convert(target.value(), runtime)?,
        ),
        Value::String(_) => {
            crate::call::argument_text::convert(target.value(), runtime)?;
            None
        }
        _ => None,
    };
    match text {
        Some(text) => {
            RootedValue::new(Value::String(text.into())).map_err(|e| e.into_runtime_error())
        }
        None => Ok(target),
    }
}

/// The existing object-receiver feval protocol. The component vector is
/// separate legacy indexing control metadata, not a copied Value workspace;
/// its complete finite allocation contract remains an explicit frontier.
pub(crate) async fn invoke_object_retained(
    runtime: Option<&RuntimeContext>,
    receiver: Value,
    arguments: BuiltinArguments,
    requested_outputs: usize,
) -> crate::BuiltinResult<RootedValueSequence> {
    if !matches!(receiver, Value::Object(_) | Value::HandleObject(_)) {
        return Err(runtime_descriptor_error_with_detail(
            "feval",
            &FEVAL_ERROR_FUNCTION_VALUE_UNSUPPORTED,
            format!("{receiver:?}"),
        ));
    }
    // Protocol construction consumes projections; retain the original physical
    // collection, capture generation, and registrations throughout the await.
    let _argument_sources = arguments.fork();
    let path = crate::object::indexing::ObjectSubscriptPath::single(
        crate::object::indexing::ObjectSubscript::parentheses(
            crate::object::indexing::ObjectIndexSelector::IndexValues {
                components: runmat_value::ExactArrayStorage::shared_unbounded(
                    arguments
                        .into_iter()
                        .map(crate::object::indexing::ObjectIndexComponent::from_protocol_value)
                        .collect::<Vec<_>>(),
                ),
            },
        ),
    );
    let request = crate::object::protocol::SubscriptReadRequest::select_outputs(
        requested_outputs,
        crate::object::protocol::ObjectAccessContext::default(),
    );
    let usage = request.sequence_use;
    let sequence = match runtime {
        Some(runtime) => {
            crate::object::protocol::read_subscript_path_sequence_with_access_in_context(
                runtime, receiver, path, request,
            )
            .await?
        }
        None => {
            crate::object::protocol::read_subscript_path_sequence_with_access(
                receiver, path, request,
            )
            .await?
        }
    };
    use crate::sequence::ResolveValueSequence;
    sequence.try_map_owned(
        |sequence| {
            let values =
                sequence.resolve(usage, crate::sequence::SequenceResolutionContext::default())?;
            if requested_outputs == 1 {
                ValueSequence::single(
                    crate::sequence::into_first_output(values)?
                        .expect("one requested output was resolved"),
                )
                .map_err(crate::sequence::sequence_error_to_runtime)
            } else {
                ValueSequence::from_output_storage(values)
                    .map_err(crate::sequence::sequence_error_to_runtime)
            }
        },
        |error| error.into_runtime_error(),
    )
}
