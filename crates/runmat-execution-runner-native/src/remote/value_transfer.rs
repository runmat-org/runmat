use std::collections::{HashMap, HashSet};

use runmat_execution::identity::ValueId;
use runmat_execution::value::{InlineValue, ValueLimits, ValuePayload, ValueRef};

use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) const MAXIMUM_REMOTE_VALUE_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn decode_value(
    reference: &ValueRef,
    encoded: &[u8],
    run_identity: &str,
) -> NativeExecutionResult<ValuePayload> {
    if reference.authorization_scope != run_identity
        || reference.encoded_length != encoded.len() as u64
        || reference.encoded_length > MAXIMUM_REMOTE_VALUE_BYTES
    {
        return Err(protocol(
            "remote value reference is outside transfer authority",
        ));
    }
    let value: ValuePayload = serde_json::from_slice(encoded).map_err(protocol)?;
    value
        .validate(ValueLimits {
            max_inline_bytes: MAXIMUM_REMOTE_VALUE_BYTES,
            ..ValueLimits::default()
        })
        .map_err(protocol)?;
    if value.logical_digest().map_err(protocol)? != reference.logical_digest {
        return Err(protocol(
            "remote value logical digest does not match its payload",
        ));
    }
    Ok(value)
}

pub(super) fn collect_references(values: &[ValuePayload]) -> Vec<ValueRef> {
    let mut references = Vec::new();
    for value in values {
        visit(value, &mut references);
    }
    references
}

pub(super) fn materialize(
    value: &ValuePayload,
    objects: &HashMap<ValueId, ValuePayload>,
) -> NativeExecutionResult<ValuePayload> {
    materialize_inner(value, objects, &mut HashSet::new(), 0)
}

fn visit(value: &ValuePayload, references: &mut Vec<ValueRef>) {
    match value {
        ValuePayload::Object(reference) => references.push((**reference).clone()),
        ValuePayload::Inline(value) => match value.as_ref() {
            InlineValue::Cell { values, .. } | InlineValue::OutputList(values) => {
                values.iter().for_each(|value| visit(value, references));
            }
            InlineValue::Struct(fields) => fields
                .iter()
                .for_each(|field| visit(&field.value, references)),
            InlineValue::Symbolic(data) | InlineValue::ImmutableValueClass(data) => data
                .fields
                .iter()
                .for_each(|field| visit(&field.value, references)),
            InlineValue::Callable(callable) => callable
                .captures
                .iter()
                .for_each(|value| visit(value, references)),
            _ => {}
        },
    }
}

fn materialize_inner(
    value: &ValuePayload,
    objects: &HashMap<ValueId, ValuePayload>,
    active: &mut HashSet<ValueId>,
    depth: u16,
) -> NativeExecutionResult<ValuePayload> {
    if depth > ValueLimits::default().max_depth {
        return Err(protocol("remote value graph exceeds maximum depth"));
    }
    match value {
        ValuePayload::Object(reference) => {
            if !active.insert(reference.id) {
                return Err(protocol("remote value graph contains a cycle"));
            }
            let stored = objects
                .get(&reference.id)
                .ok_or_else(|| protocol("remote value object is unavailable"))?;
            let value = materialize_inner(stored, objects, active, depth + 1)?;
            active.remove(&reference.id);
            Ok(value)
        }
        ValuePayload::Inline(inline) => {
            let mut inline = (**inline).clone();
            match &mut inline {
                InlineValue::Cell { values, .. } | InlineValue::OutputList(values) => {
                    for value in values {
                        *value = materialize_inner(value, objects, active, depth + 1)?;
                    }
                }
                InlineValue::Struct(fields) => {
                    for field in fields {
                        field.value = materialize_inner(&field.value, objects, active, depth + 1)?;
                    }
                }
                InlineValue::Symbolic(data) | InlineValue::ImmutableValueClass(data) => {
                    for field in &mut data.fields {
                        field.value = materialize_inner(&field.value, objects, active, depth + 1)?;
                    }
                }
                InlineValue::Callable(callable) => {
                    for capture in &mut callable.captures {
                        *capture = materialize_inner(capture, objects, active, depth + 1)?;
                    }
                }
                _ => {}
            }
            Ok(ValuePayload::Inline(Box::new(inline)))
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
