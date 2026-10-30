//! Canonical graphics predicate; candidates remain rooted in BuiltinArguments.
use runmat_builtins::{
    ISGRAPHICS_DESCRIPTOR, ISGRAPHICS_ERROR_INVALID_ARGUMENT, ISGRAPHICS_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Value};
#[path = "isgraphics/selector.rs"]
mod selector;
#[cfg(test)]
#[path = "isgraphics/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "isgraphics",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::isgraphics"
)]
pub fn isgraphics_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &ISGRAPHICS_DESCRIPTOR,
        &ISGRAPHICS_ERROR_TOO_MANY_OUTPUTS,
        "isgraphics",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "isgraphics")?;
    work.finish((|| {
        if !(1..=2).contains(&args.len()) {
            return Err(invalid_argument(
                "expected one candidate and optional type text",
            ));
        }
        let kind = args.get(1).map(graphics_type_name).transpose()?;
        let result = handle_predicate_value(&args[0], "isgraphics");
        Ok(match kind.as_ref() {
            Some(kind) => handle_predicate_value_of_type(&args[0], kind, result),
            None => result,
        })
    })())
}
fn invalid_argument(detail: &str) -> crate::RuntimeError {
    super::coordinate_admission::terminal(crate::runtime_descriptor_error_with_detail(
        "isgraphics",
        &ISGRAPHICS_ERROR_INVALID_ARGUMENT,
        detail,
    ))
}
fn graphics_type_name(value: &Value) -> crate::BuiltinResult<selector::GraphicsType> {
    let text = match value {
        Value::String(runmat_value::StringElement::Text(value)) => {
            value.trim().to_ascii_lowercase()
        }
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            value.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .trim()
                .to_ascii_lowercase()
        }
        Value::CharArray(value) if value.rows == 1 => {
            runmat_value::decode_utf16_code_units(value.data())
                .map_err(|source| {
                    let mut error = invalid_argument(&source.to_string());
                    error.source = Some(Box::new(source));
                    error
                })?
                .trim()
                .to_ascii_lowercase()
        }
        _ => return Err(invalid_argument("type must be a host text scalar")),
    };
    Ok(selector::GraphicsType::parse(&text))
}

fn handle_predicate_value_of_type(
    value: &Value,
    kind: &selector::GraphicsType,
    base: Value,
) -> Value {
    let scalar_matches = |value: &Value| graphics_handle_matches_type(value, kind);
    match (value, base) {
        (Value::Tensor(tensor), Value::LogicalArray(mut logical)) => {
            for index in 0..tensor.len() {
                let candidate = tensor
                    .numeric_value_at(index)
                    .and_then(numeric_graphics_handle)
                    .map(Value::Num)
                    .unwrap_or(Value::Num(f64::NAN));
                logical.data[index] &= u8::from(scalar_matches(&candidate));
            }
            Value::LogicalArray(logical)
        }
        (_, Value::Bool(base)) => Value::Bool(base && scalar_matches(value)),
        (_, other) => other,
    }
}

fn numeric_graphics_handle(value: runmat_value::NumericScalar) -> Option<f64> {
    use runmat_value::NumericScalar;
    match value {
        NumericScalar::F64(value) => Some(value),
        NumericScalar::F32(value) => Some(f64::from(value)),
        integer => integer.into_int_value().and_then(|integer| {
            crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(&integer)
        }),
    }
}

fn graphics_handle_matches_type(value: &Value, kind: &selector::GraphicsType) -> bool {
    graphics_handle_valid(value, Some(kind), "isgraphics")
}
pub(super) fn graphics_handle_valid(
    value: &Value,
    kind: Option<&selector::GraphicsType>,
    builtin: &'static str,
) -> bool {
    if let Value::HandleObject(handle) = value {
        if !crate::is_handle_valid(handle) {
            return false;
        }
        return runmat_gc::gc_with_value(&handle.target, |target| {
            let Value::Object(object) = target else {
                return false;
            };
            object
                .graphics_receiver()
                .and_then(crate::builtins::plotting::state::graphics_receiver_class)
                .filter(|identity| handle.class_name == *identity)
                .is_some_and(|identity| kind.is_none_or(|kind| kind.matches_class(identity)))
        })
        .unwrap_or(false);
    }
    crate::builtins::plotting::properties::resolve_plot_handle(value, builtin)
        .is_ok_and(|handle| kind.is_none_or(|kind| kind.matches_handle(&handle)))
}

fn handle_predicate_value(value: &Value, builtin: &'static str) -> Value {
    match value {
        Value::GpuTensor(handle) => false_value(handle.shape().to_vec()),
        Value::Int(integer) => {
            if !crate::compatibility::runmat_extensions_enabled() {
                return Value::Bool(false);
            }
            Value::Bool(
                crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(integer)
                    .is_some_and(|handle| handle_is_graphics(handle, builtin)),
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if crate::compatibility::ensure_builtin_extension_enabled(
                &runmat_builtins::ISGRAPHICS_INTEGER_HANDLE_EXTENSION,
                builtin,
            )
            .is_err()
            {
                return Value::LogicalArray(
                    LogicalArray::new(vec![0; tensor.len()], tensor.shape().to_vec())
                        .expect("logical shape from tensor"),
                );
            }
            integer_handle_predicate(tensor, builtin, handle_is_graphics)
        }
        Value::Tensor(tensor) => numeric_handle_predicate(tensor, builtin, handle_is_graphics),
        _ => Value::Bool(graphics_handle_valid(value, None, builtin)),
    }
}

fn false_value(shape: Vec<usize>) -> Value {
    if shape.iter().product::<usize>() == 1 {
        Value::Bool(false)
    } else {
        Value::LogicalArray(
            LogicalArray::new(vec![0; shape.iter().product()], shape)
                .expect("logical shape from input metadata"),
        )
    }
}

fn integer_handle_predicate(
    tensor: &runmat_value::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let storage = tensor.integer_storage().expect("integer handle storage");
    let data =
        (0..storage.len())
            .map(|index| {
                u8::from(
        storage.value_at(index).as_ref()
            .and_then(crate::builtins::plotting::op_common::handles::numeric_handle_from_integer)
            .is_some_and(|handle| predicate(handle, builtin))
    )
            })
            .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape().to_vec()).expect("logical shape from tensor"),
    )
}

fn numeric_handle_predicate(
    tensor: &runmat_value::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let data = (0..tensor.len())
        .map(|index| {
            u8::from(predicate(
                crate::builtins::common::tensor::tensor_value_f64(tensor, index),
                builtin,
            ))
        })
        .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape().to_vec()).expect("logical shape from tensor"),
    )
}

fn handle_is_graphics(handle: f64, builtin: &'static str) -> bool {
    if !handle.is_finite() || handle < 0.0 {
        return false;
    }
    crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), builtin).is_ok()
}
