use runmat_value::ExactArrayIntoIter;

use runmat_value::Value;

use super::{resolve_object_protocol, ObjectProtocol};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueSequence};
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::sequence::ValueSequence;
use crate::RuntimeError;

mod callable;
mod default_indexing;
mod dotted_invoke;
mod owned;
mod request;
mod selected_paren;
use default_indexing::{assign_default_steps, read_default_step, read_default_step_sequence};
pub use owned::{
    execute_owned_subsasgn, execute_owned_subsasgn_in_context, execute_owned_subsasgn_with_context,
    execute_owned_subsref, execute_owned_subsref_in_context, execute_owned_subsref_with_context,
};
pub use request::SubscriptReadRequest;

pub fn read_subscript_path<'a>(
    base: Value,
    path: ObjectSubscriptPath,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
    super::context::admit_future_with(None, move || async move {
        let request = SubscriptReadRequest::require_single(access);
        let sequence =
            read_subscript_path_sequence(None, base, path, request, caller_function_name).await?;
        resolve_single(sequence)
    })
}

pub fn read_subscript_path_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
    super::context::admit_future_with(Some(runtime), move || async move {
        let request = SubscriptReadRequest::require_single(access);
        let sequence =
            read_subscript_path_sequence(Some(runtime), base, path, request, caller_function_name)
                .await?;
        resolve_single(sequence)
    })
}

pub fn read_subscript_path_with_access(
    base: Value,
    path: ObjectSubscriptPath,
    access: super::ObjectAccessContext,
) -> crate::execution::RuntimeContinuationFuture<'static, RootedValue> {
    super::context::admit_future_with(None, move || async move {
        let sequence = read_subscript_path_sequence(
            None,
            base,
            path,
            SubscriptReadRequest::require_single(access),
            None,
        )
        .await?;
        resolve_single(sequence)
    })
}

pub fn read_subscript_path_with_access_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    access: super::ObjectAccessContext,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    super::context::admit_future_with(Some(runtime), move || async move {
        let sequence = read_subscript_path_sequence(
            Some(runtime),
            base,
            path,
            SubscriptReadRequest::require_single(access),
            None,
        )
        .await?;
        resolve_single(sequence)
    })
}

pub fn read_subscript_path_sequence_with_access(
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
) -> crate::execution::RuntimeContinuationFuture<'static, RootedValueSequence> {
    read_subscript_path_sequence(None, base, path, request, None)
}

pub fn read_subscript_path_sequence_with_access_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValueSequence> {
    read_subscript_path_sequence(Some(runtime), base, path, request, None)
}

fn read_subscript_path_sequence<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValueSequence> {
    super::context::admit_future_with(runtime, move || async move {
        let _base_root = crate::execution::RootedValueLease::for_values(std::iter::once(&base))
            .map_err(|error| error.into_runtime_error())?;
        match resolve_object_protocol(&base, ObjectProtocol::Subsref, &request.access)? {
            Some(method) => invoke_prepared_subsref(runtime, base, path, method, &request).await,
            None => {
                read_default_steps(
                    runtime,
                    base,
                    path.into_steps_in_context(runtime)?.into_iter(),
                    request,
                    caller_function_name,
                )
                .await
            }
        }
    })
}

pub fn assign_subscript_path<'a>(
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    assign_subscript_path_with_context(None, base, path, values, caller_function_name)
}

pub fn assign_subscript_path_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    assign_subscript_path_with_context(Some(runtime), base, path, values, caller_function_name)
}

fn assign_subscript_path_with_context<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
    super::context::admit_future_with(runtime, move || async move {
        let _inputs_root = crate::execution::RootedValueLease::for_values(
            std::iter::once(&base).chain(values.iter()),
        )
        .map_err(|error| error.into_runtime_error())?;
        let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
        match resolve_object_protocol(&base, ObjectProtocol::Subsasgn, &access)? {
            Some(method) => match runtime {
                Some(runtime) => {
                    super::invoke_resolved_object_assignment_in_context(
                        runtime, &method, base, path, values,
                    )
                    .await
                }
                None => super::invoke_resolved_object_assignment(&method, base, path, values).await,
            },
            None => {
                assign_default_steps(
                    runtime,
                    base,
                    path.into_steps_in_context(runtime)?.into_iter(),
                    values,
                    caller_function_name,
                )
                .await
            }
        }
    })
}

fn read_default_steps<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    mut steps: ExactArrayIntoIter<ObjectSubscript>,
    request: SubscriptReadRequest,
    caller_function_name: Option<&'a str>,
) -> crate::execution::RuntimeContinuationFuture<'a, RootedValueSequence> {
    super::context::admit_future_with(runtime, move || async move {
        let mut base = RootedValue::new(base).map_err(|error| error.into_runtime_error())?;
        while steps.len() != 0 {
            if let Some(method) =
                resolve_object_protocol(base.value(), ObjectProtocol::Subsref, &request.access)?
            {
                let remaining = ObjectSubscriptPath::from_remaining_in_context(runtime, steps)?;
                return invoke_prepared_subsref(
                    runtime,
                    base.value().clone(),
                    remaining,
                    method,
                    &request,
                )
                .await;
            }
            if let Some(sequence) = dotted_invoke::try_execute(
                runtime,
                base.value().clone(),
                &mut steps,
                &request,
                caller_function_name,
            )
            .await?
            {
                if steps.len() == 0 {
                    return Ok(sequence);
                }
                base = resolve_single(sequence)?;
                continue;
            }
            // A prior brace/member read may have selected a callable value.
            // Its following parentheses are an invocation, not numeric
            // indexing of the handle. The selected receiver is never read or
            // invoked again after this boundary.
            if callable::is_callable_value(base.value())
                && steps.as_slice().first().is_some_and(|step| {
                    step.kind() == crate::object::indexing::ObjectIndexKind::Paren
                })
            {
                let step = steps.next().ok_or_else(|| {
                    semantic_error(
                        "InvalidObjectSubscriptPath",
                        "object subscript path is empty",
                    )
                })?;
                let requested_outputs = if steps.len() == 0 {
                    request.known_requested_outputs()?.ok_or_else(|| {
                        semantic_error(
                            "SequenceOutputContextUnavailable",
                            "callable invocation requires an explicit output count",
                        )
                    })?
                } else {
                    1
                };
                let arguments = callable::paren_arguments(runtime, step)?;
                let sequence = base
                    .with_owned_async(|target| {
                        callable::invoke(runtime, target, arguments.into(), requested_outputs)
                    })
                    .await?;
                if steps.len() == 0 {
                    return Ok(sequence);
                }
                base = resolve_single(sequence)?;
                continue;
            }
            if selected_paren::supports(base.value(), steps.as_slice().first()) {
                let step = steps.next().ok_or_else(|| {
                    semantic_error(
                        "InvalidObjectSubscriptPath",
                        "object subscript path is empty",
                    )
                })?;
                let value = base
                    .with_owned_async(|target| selected_paren::execute(runtime, target, step))
                    .await?;
                if steps.len() == 0 {
                    return value.try_map_owned(
                        |value| {
                            ValueSequence::single(value)
                                .map_err(crate::sequence::sequence_error_to_runtime)
                        },
                        |error| error.into_runtime_error(),
                    );
                }
                base = value;
                continue;
            }
            let step = steps.next().ok_or_else(|| {
                semantic_error(
                    "InvalidObjectSubscriptPath",
                    "object subscript path is empty",
                )
            })?;
            if steps.len() == 0 {
                return read_default_step_sequence(
                    runtime,
                    base.value().clone(),
                    &step,
                    caller_function_name,
                )
                .await;
            }
            base = read_default_step(runtime, base.value().clone(), &step, caller_function_name)
                .await?;
        }
        base.try_map_owned(
            |base| ValueSequence::single(base).map_err(crate::sequence::sequence_error_to_runtime),
            |error| error.into_runtime_error(),
        )
    })
}

async fn invoke_prepared_subsref(
    runtime: Option<&RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    method: super::ResolvedObjectMethod,
    request: &SubscriptReadRequest,
) -> Result<RootedValueSequence, RuntimeError> {
    let requested_outputs = match request.known_requested_outputs()? {
        Some(count) => count,
        None => match runtime {
            Some(runtime) => {
                crate::builtins::introspection::object_indexing::object_path_cardinality_in_context(
                    runtime,
                    &base,
                    &path,
                    request.indexing_context,
                    &request.access,
                )
                .await?
                .count
            }
            None => {
                crate::builtins::introspection::object_indexing::object_path_cardinality(
                    &base,
                    &path,
                    request.indexing_context,
                    &request.access,
                )
                .await?
                .count
            }
        },
    };
    let resolution = Some(method);
    match runtime {
        Some(runtime) => {
            super::invoke_resolved_object_protocol_in_context(
                runtime,
                &resolution,
                base,
                path,
                requested_outputs,
            )
            .await
        }
        None => {
            super::invoke_resolved_object_protocol(&resolution, base, path, requested_outputs).await
        }
    }
}

fn resolve_single(sequence: RootedValueSequence) -> Result<RootedValue, RuntimeError> {
    sequence.try_into_single()
}
