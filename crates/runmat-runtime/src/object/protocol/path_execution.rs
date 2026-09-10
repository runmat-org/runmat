use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use runmat_value::Value;

use super::{resolve_object_protocol, ObjectProtocol, ProtocolResolution};
use crate::object::indexing::{ObjectSubscript, ObjectSubscriptPath};
use crate::runtime_error::semantic_error;
use crate::sequence::{SequenceResolutionContext, ValueSequence};
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
            read_subscript_path_sequence(base, path, request, caller_function_name).await?;
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
    read_subscript_path_sequence(base, path, request, None)
}

fn read_subscript_path_sequence<'a>(
    base: Value,
    path: ObjectSubscriptPath,
    request: SubscriptReadRequest,
    caller_function_name: Option<&'a str>,
) -> Pin<Box<dyn Future<Output = Result<ValueSequence, RuntimeError>> + 'a>> {
    Box::pin(async move {
        match resolve_object_protocol(&base, ObjectProtocol::Subsref, &request.access)? {
            ProtocolResolution::Method(method) => {
                invoke_prepared_subsref(base, path, method, &request).await
            }
            ProtocolResolution::DefaultIndexing => {
                read_default_steps(
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
    Box::pin(async move {
        let access = super::ObjectAccessContext::from_legacy_function_name(caller_function_name);
        match resolve_object_protocol(&base, ObjectProtocol::Subsasgn, &access)? {
            ProtocolResolution::Method(method) => {
                super::invoke_resolved_object_assignment(&method, base, path, values).await
            }
            ProtocolResolution::DefaultIndexing => {
                assign_default_steps(base, path.into_steps().into(), values, caller_function_name)
                    .await
            }
        }
    })
}

fn read_default_steps<'a>(
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
                return invoke_prepared_subsref(base, remaining, method, &request).await;
            }
            if let Some(sequence) =
                dotted_invoke::try_execute(base.clone(), &mut steps, &request, caller_function_name)
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
                return read_default_step_sequence(base, &step, caller_function_name).await;
            }
            base = read_default_step(base, &step, caller_function_name).await?;
        }
        Ok(ValueSequence::single(base))
    })
}

async fn invoke_prepared_subsref(
    base: Value,
    path: ObjectSubscriptPath,
    method: super::ResolvedObjectMethod,
    request: &SubscriptReadRequest,
) -> Result<ValueSequence, RuntimeError> {
    let requested_outputs = match request.known_requested_outputs()? {
        Some(count) => count,
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
    };
    let value = super::invoke_resolved_object_protocol(
        &ProtocolResolution::Method(method),
        base,
        path,
        requested_outputs,
    )
    .await?;
    Ok(ValueSequence::from_callable_result(
        value,
        requested_outputs,
    ))
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
