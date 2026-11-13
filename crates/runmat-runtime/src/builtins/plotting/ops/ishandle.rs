//! Canonical graphics predicate; candidates remain rooted in BuiltinArguments.
use runmat_builtins::{
    ISHANDLE_DESCRIPTOR, ISHANDLE_ERROR_INVALID_ARGUMENT, ISHANDLE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Value};

#[cfg(test)]
#[path = "ishandle/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "ishandle",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::ishandle"
)]
pub fn ishandle_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &ISHANDLE_DESCRIPTOR,
        &ISHANDLE_ERROR_TOO_MANY_OUTPUTS,
        "ishandle",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "ishandle")?;
    work.finish((|| {
        if args.len() != 1 {
            return Err(invalid_argument("expected exactly one candidate"));
        }

        let result = handle_predicate_value(&args[0], "ishandle");
        Ok(result)
    })())
}
fn invalid_argument(detail: &str) -> crate::RuntimeError {
    super::coordinate_admission::terminal(crate::runtime_descriptor_error_with_detail(
        "ishandle",
        &ISHANDLE_ERROR_INVALID_ARGUMENT,
        detail,
    ))
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
                    .is_some_and(|handle| handle_is_valid(handle, builtin)),
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if crate::compatibility::ensure_builtin_extension_enabled(
                &runmat_builtins::ISHANDLE_INTEGER_HANDLE_EXTENSION,
                builtin,
            )
            .is_err()
            {
                return Value::LogicalArray(
                    LogicalArray::new(vec![0; tensor.len()], tensor.shape().to_vec())
                        .expect("logical shape from tensor"),
                );
            }
            integer_handle_predicate(tensor, builtin, handle_is_valid)
        }
        Value::Tensor(tensor) => numeric_handle_predicate(tensor, builtin, handle_is_valid),
        _ => Value::Bool(super::isgraphics::graphics_handle_valid(
            value, None, builtin,
        )),
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

fn handle_is_valid(handle: f64, builtin: &'static str) -> bool {
    if !handle.is_finite() || handle < 0.0 {
        return false;
    }
    crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), builtin).is_ok()
}
