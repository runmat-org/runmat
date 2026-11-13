use super::op_common::current_axes_target;
use super::op_common::AxesTarget;
use super::state::{set_view_for_axes, FigureError};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, VIEW_ERROR_INTERNAL, VIEW_ERROR_INVALID_ARGUMENT,
    VIEW_ERROR_INVALID_AXES, VIEW_ERROR_TOO_MANY_OUTPUTS, VIEW_INTEGER_ANGLES_EXTENSION,
    VIEW_INTEGER_AXES_EXTENSION, VIEW_VECTOR_OUTPUT_EXTENSION,
};
use runmat_value::{IntValue, IntegerStorage, NumericScalar, Tensor, Value};
const BUILTIN_NAME: &str = "view";
fn view_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    view_error_with_message(error.message, error)
}

fn view_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_view_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidSubplotIndex { .. }
        | FigureError::InvalidAxesHandle
        | FigureError::InvalidHandle(_) => view_error(&VIEW_ERROR_INVALID_AXES),
        other => view_error_with_message(
            format!("{}: {}", VIEW_ERROR_INTERNAL.message, other),
            &VIEW_ERROR_INTERNAL,
        ),
    }
}

fn parse_view_target(
    args: crate::call::FunctionArgumentView<'_>,
) -> (AxesTarget, crate::call::FunctionArgumentView<'_>) {
    if let Some(first) = args.first() {
        if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(fig, axes)) =
            crate::builtins::plotting::properties::resolve_plot_handle(first, "view")
        {
            return (
                Some((fig, axes)),
                args.range(1..).expect("nonempty view target"),
            );
        }
    }
    (None, args)
}

fn parse_view_angles(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<(f32, f32)> {
    match args.len() {
        1 => {
            let tensor = scalar_or_tensor(&args[0])?;
            if tensor.len() == 1 {
                let value = view_value_f32(&tensor, 0)?;
                if value == 2.0 {
                    Ok((0.0, 90.0))
                } else if value == 3.0 {
                    Ok((-37.5, 30.0))
                } else {
                    Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT))
                }
            } else if tensor.len() == 2 {
                Ok((view_value_f32(&tensor, 0)?, view_value_f32(&tensor, 1)?))
            } else {
                Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT))
            }
        }
        2 => {
            let az = scalar_or_tensor(&args[0])?;
            let el = scalar_or_tensor(&args[1])?;
            if !tensor::is_scalar_tensor(&az) || !tensor::is_scalar_tensor(&el) {
                return Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT));
            }
            Ok((view_value_f32(&az, 0)?, view_value_f32(&el, 0)?))
        }
        _ => Err(view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

fn scalar_or_tensor(value: &Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::Num(v) => {
            Tensor::new(vec![*v], vec![1, 1]).map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT))
        }
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i.clone()), vec![1, 1])
            .map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
        other => Tensor::try_from(other).map_err(|_| view_error(&VIEW_ERROR_INVALID_ARGUMENT)),
    }
}

fn view_value_f32(tensor: &Tensor, index: usize) -> crate::BuiltinResult<f32> {
    let value = tensor
        .numeric_value_at(index)
        .ok_or_else(|| view_error(&VIEW_ERROR_INVALID_ARGUMENT))?;
    match value {
        NumericScalar::F64(value) => finite_angle(value),
        NumericScalar::F32(value) => finite_angle(value as f64),
        integer => {
            let integer = integer
                .into_int_value()
                .expect("nonfloating NumericScalar is an integer");
            exact_integer_f32(integer).ok_or_else(|| {
                view_error_with_message(
                    "view: integer angles must be exactly representable as single",
                    &VIEW_ERROR_INVALID_ARGUMENT,
                )
            })
        }
    }
}

fn finite_angle(value: f64) -> crate::BuiltinResult<f32> {
    super::properties::finite_camera_angle(value).ok_or_else(|| {
        view_error_with_message(
            "view: angles must be finite in single-precision camera state",
            &VIEW_ERROR_INVALID_ARGUMENT,
        )
    })
}

fn exact_integer_f32(value: IntValue) -> Option<f32> {
    use num_traits::FromPrimitive;
    let converted = value.to_f64() as f32;
    let exact = match value {
        IntValue::I8(value) => num_bigint::BigInt::from(value),
        IntValue::I16(value) => num_bigint::BigInt::from(value),
        IntValue::I32(value) => num_bigint::BigInt::from(value),
        IntValue::I64(value) => num_bigint::BigInt::from(value),
        IntValue::U8(value) => num_bigint::BigInt::from(value),
        IntValue::U16(value) => num_bigint::BigInt::from(value),
        IntValue::U32(value) => num_bigint::BigInt::from(value),
        IntValue::U64(value) => num_bigint::BigInt::from(value),
    };
    (num_bigint::BigInt::from_f32(converted).as_ref() == Some(&exact)).then_some(converted)
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

#[runmat_macros::runtime_builtin(
    name = "view",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::view"
)]
pub fn view_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 2) {
        return Err(view_error(&VIEW_ERROR_TOO_MANY_OUTPUTS));
    }
    if outputs == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &VIEW_VECTOR_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let Some(first) = args.first() {
        if is_typed_integer(first)
            && matches!(
                crate::builtins::plotting::properties::resolve_plot_handle(first, BUILTIN_NAME),
                Ok(crate::builtins::plotting::properties::PlotHandle::Axes(
                    _,
                    _
                ))
            )
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &VIEW_INTEGER_AXES_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    let (target, rest) = parse_view_target(args.view());
    if rest.is_empty() {
        let target = target.unwrap_or_else(current_axes_target);
        let meta = crate::builtins::plotting::state::axes_metadata_snapshot(target.0, target.1)
            .map_err(map_view_figure_error)?;
        let az = meta.view_azimuth_deg.unwrap_or(-37.5) as f64;
        let el = meta.view_elevation_deg.unwrap_or(30.0) as f64;
        return Ok(view_outputs(az, el, outputs));
    }
    if rest.iter().any(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &VIEW_INTEGER_ANGLES_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (az, el) = parse_view_angles(rest)?;
    let target = target.unwrap_or_else(current_axes_target);
    set_view_for_axes(target.0, target.1, az, el).map_err(map_view_figure_error)?;
    Ok(view_outputs(az as f64, el as f64, outputs))
}

fn view_outputs(az: f64, el: f64, outputs: Option<usize>) -> Value {
    match outputs {
        Some(0) => Value::output_list(vec![]),
        Some(2) => Value::output_list(vec![Value::Num(az), Value::Num(el)]),
        _ => Value::Tensor(Tensor::new(vec![az, el], vec![1, 2]).expect("view vector")),
    }
}

#[cfg(test)]
#[path = "view/tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) fn view_builtin_source(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    view_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?)
}
