//! Complete constructor field snapshots before publishing a value or handle.
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{build_runtime_error, RuntimeError};
use runmat_value::{ObjectInstance, StructValue, Value};

pub(super) fn normalize(
    runtime: Option<&RuntimeContext>,
    default: Value,
    result: Value,
    requested_outputs: usize,
) -> Result<RootedValue, RuntimeError> {
    if requested_outputs != 1 {
        return RootedValue::in_optional_context(result, runtime);
    }
    let value = match result {
        Value::Struct(fields) => match default {
            Value::Object(mut object) => {
                merge(runtime, &mut object, &fields)?;
                Value::Object(object)
            }
            Value::HandleObject(handle) => {
                let snapshot =
                    runmat_gc::gc_with_value(&handle.target, |value| object(value).cloned())
                        .map_err(gc_error)??;
                let mut replacement = snapshot.clone();
                merge(runtime, &mut replacement, &fields)?;
                let replacement =
                    RootedValue::in_optional_context(Value::Object(replacement), runtime)?;
                let prepared = replacement.into_value_and_lease();
                let _roots = prepared.1;
                let replacement = prepared.0;
                let old = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                    if !object(target)?.shares_generation_with(&snapshot) {
                        return Err(build_runtime_error(
                            "constructor target changed before publication",
                        )
                        .with_identifier("RunMat:ConstructorOwnerChanged")
                        .build());
                    }
                    runmat_gc::gc_record_handle_write(&handle.target, &replacement);
                    Ok(std::mem::replace(target, replacement))
                })
                .map_err(gc_error)??;
                drop(old);
                Value::HandleObject(handle)
            }
            _ => Value::Struct(fields),
        },
        value @ (Value::Object(_) | Value::HandleObject(_)) => value,
        _ => default,
    };
    RootedValue::in_optional_context(value, runtime)
}

fn object(value: &Value) -> Result<&ObjectInstance, RuntimeError> {
    match value {
        Value::Object(object) if crate::object_handle_flag_valid(object) => Ok(object),
        Value::Object(_) => {
            Err(build_runtime_error("constructor result handle target is invalid").build())
        }
        _ => Err(build_runtime_error("constructor result handle target is not an object").build()),
    }
}

fn merge(
    runtime: Option<&RuntimeContext>,
    object: &mut ObjectInstance,
    fields: &StructValue,
) -> Result<(), RuntimeError> {
    crate::execution::object::replacement::replace_properties(
        runtime,
        object,
        fields.fields.iter(),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:Constructor:Fields",
                message,
                source,
            )
        },
        |message| {
            build_runtime_error(message)
                .with_identifier("RunMat:Constructor:Fields")
                .build()
        },
    )
}

fn gc_error(source: runmat_gc::GcError) -> RuntimeError {
    build_runtime_error(format!(
        "constructor result handle target invalid: {source}"
    ))
    .with_source(source)
    .build()
}
