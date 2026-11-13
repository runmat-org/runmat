//! MATLAB-compatible `superclasses` builtin backed by RunMat class metadata.
use runmat_builtins::{
    BuiltinErrorDescriptor, SUPERCLASSES_ERROR_CLASS_INVALID, SUPERCLASSES_ERROR_CLASS_UNKNOWN,
};
const BUILTIN_NAME: &str = "superclasses";
use crate::builtins::introspection::class_identity::class_identity_for_value;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Value};

#[runtime_builtin(
    name = "superclasses",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::superclasses"
)]
fn superclasses_builtin(value: Value) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SUPERCLASSES_DESCRIPTOR,
        &runmat_builtins::SUPERCLASSES_ERROR_OUTPUTS,
        "superclasses",
    )?;
    let class_name = requested_class_name(&value)?;
    let supers = if let Some(chain) = crate::class_registry::superclass_chain(&class_name) {
        chain
    } else if known_leaf_class_without_registered_metadata(&class_name) {
        Vec::new()
    } else {
        return Err(superclasses_error_with_detail(
            &SUPERCLASSES_ERROR_CLASS_UNKNOWN,
            class_name.display_name().to_owned(),
        ));
    };

    let cols = supers.len();
    let cells = supers
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(name.display_name())))
        .collect::<Vec<_>>();
    Ok(Value::Cell(
        CellArray::new(cells, 1, cols).map_err(superclasses_message_error)?,
    ))
}

fn requested_class_name(value: &Value) -> BuiltinResult<runmat_types::ClassIdentity> {
    let source = match value {
        Value::String(runmat_value::StringElement::Text(class_name)) => {
            class_name.as_str().to_owned()
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => sa.data()
            [0]
        .text_ref()
        .expect("checked nonmissing scalar string")
        .as_str()
        .to_owned(),
        Value::CharArray(ca) if ca.rows <= 1 => {
            super::text::decode_utf16_with_error(ca.data(), || {
                superclasses_error(&SUPERCLASSES_ERROR_CLASS_INVALID)
            })?
        }
        Value::ClassRef(class_name) => return Ok(class_name.as_identity().clone()),
        Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::GpuTensor(_)
        | Value::Listener(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_) => return Ok(class_identity_for_value(value)),
        _ => return Err(superclasses_error(&SUPERCLASSES_ERROR_CLASS_INVALID)),
    };
    runmat_types::ClassIdentity::new(source).map_err(|error| {
        superclasses_error_with_detail(&SUPERCLASSES_ERROR_CLASS_UNKNOWN, error.to_string())
    })
}

fn known_leaf_class_without_registered_metadata(class_name: &runmat_types::ClassIdentity) -> bool {
    use runmat_types::standard;
    [
        standard::DOUBLE,
        standard::SINGLE,
        standard::LOGICAL,
        standard::CHAR,
        standard::STRING,
        standard::CELL,
        standard::STRUCT,
        standard::FUNCTION_HANDLE,
        standard::HANDLE,
        standard::GPU_ARRAY,
        standard::META_CLASS,
        standard::MEXCEPTION,
        standard::EVENT_LISTENER,
        standard::SPARSE,
        standard::SYMBOLIC,
        standard::INT8,
        standard::UINT8,
        standard::INT16,
        standard::UINT16,
        standard::INT32,
        standard::UINT32,
        standard::INT64,
        standard::UINT64,
    ]
    .into_iter()
    .any(|identity| class_name.is(identity))
}

fn superclasses_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn superclasses_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: String,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn superclasses_message_error(message: String) -> RuntimeError {
    build_runtime_error(format!("superclasses: {message}"))
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[cfg(test)]
mod tests;
