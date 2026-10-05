use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

use super::{resolve_object_protocol, ObjectProtocol, ProtocolResolution};
use crate::context::RuntimeContext;
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::sequence::{ResolveValueSequence, SequenceResolutionContext, ValueSequence};
use crate::RuntimeError;

mod default_indexing;
mod dotted_invoke;
mod owned;
mod request;
use default_indexing::{assign_default_steps, read_default_step, read_default_step_sequence};
pub use owned::{execute_owned_subsasgn, execute_owned_subsref};
pub use request::SubscriptReadRequest;

pub fn read_subscript_path<'a>(
    base: Value,
    path: ObjectSubscriptPath,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
    Box::pin(async move {
        let request = SubscriptReadRequest::require_single(access);
        let sequence =
            read_subscript_path_sequence(None, base, path, request, caller_function_name).await?;
        resolve_single(sequence)
    })
}

pub fn read_subscript_path_with_access(
    base: Value,
    path: ObjectSubscriptPath,
    access: super::ObjectAccessContext,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>>>> {
    Box::pin(async move {
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
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
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
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>>>> {
    read_subscript_path_sequence(None, base, path, request, None)
}

pub fn read_subscript_path_sequence_with_access_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>> + 'a>> {
    read_subscript_path_sequence(Some(runtime), base, path, request, None)
}

fn read_subscript_path_sequence<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>> + 'a>> {
    Box::pin(async move {
        match resolve_object_protocol(&base, ObjectProtocol::Subsref, &request.access)? {
            ProtocolResolution::Method(method) => {
                invoke_prepared_subsref(runtime, base, path, method, &request).await
            }
            ProtocolResolution::DefaultIndexing => {
                read_default_steps(
                    runtime,
                    base,
                    path.into_steps().into(),
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
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    assign_subscript_path_with_context(None, base, path, values, caller_function_name)
}

pub fn assign_subscript_path_in_context<'a>(
    runtime: &'a RuntimeContext,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    assign_subscript_path_with_context(Some(runtime), base, path, values, caller_function_name)
}

fn assign_subscript_path_with_context<'a>(
    runtime: Option<&'a RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    values: Vec<Value>,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
        match resolve_object_protocol(&base, ObjectProtocol::Subsasgn, &access)? {
            ProtocolResolution::Method(method) => match runtime {
                Some(runtime) => {
                    super::invoke_resolved_object_assignment_in_context(
                        runtime, &method, base, path, values,
                    )
                    .await
                }
                None => super::invoke_resolved_object_assignment(&method, base, path, values).await,
            },
            ProtocolResolution::DefaultIndexing => {
                assign_default_steps(
                    runtime,
                    base,
                    path.into_steps().into(),
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
    mut base: Value,
    mut steps: VecDeque<ObjectSubscript>,
    request: SubscriptReadRequest,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>> + 'a>> {
    Box::pin(async move {
        while !steps.is_empty() {
            if let ProtocolResolution::Method(method) =
                resolve_object_protocol(&base, ObjectProtocol::Subsref, &request.access)?
            {
                let remaining = ObjectSubscriptPath::new(steps.into_iter().collect())?;
                return invoke_prepared_subsref(runtime, base, remaining, method, &request).await;
            }
            if let Some(sequence) = dotted_invoke::try_execute(
                runtime,
                base.clone(),
                &mut steps,
                &request,
                caller_function_name,
            )
            .await?
            {
                if steps.is_empty() {
                    return Ok(sequence);
                }
                base = resolve_single(sequence)?;
                continue;
            }
            let step = steps.pop_front().ok_or_else(|| {
                semantic_error(
                    "InvalidObjectSubscriptPath",
                    "object subscript path is empty",
                )
            })?;
            if steps.is_empty() {
                return read_default_step_sequence(runtime, base, &step, caller_function_name)
                    .await;
            }
            base = read_default_step(runtime, base, &step, caller_function_name).await?;
        }
        ValueSequence::single(base).map_err(crate::sequence::sequence_error_to_runtime)
    })
}

async fn invoke_prepared_subsref(
    runtime: Option<&RuntimeContext>,
    base: Value,
    path: ObjectSubscriptPath,
    method: super::ResolvedObjectMethod,
    request: &SubscriptReadRequest,
) -> Result<ValueSequence, RuntimeError> {
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
    let resolution = ProtocolResolution::Method(method);
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

fn resolve_single(sequence: ValueSequence) -> Result<Value, RuntimeError> {
    let mut values = sequence.resolve(
        runmat_types::SequenceUse::RequireSingle,
        SequenceResolutionContext::default(),
    )?;
    values.pop().ok_or_else(|| {
        semantic_error(
            "CommaSeparatedListRequiresSingleValue",
            "subscript read requires one value",
        )
    })
}
