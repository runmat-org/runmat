//! Host-visible Q-Q plot roles and class/shape checks before provider work.

use super::{invalid, NAME};
use crate::builtins::common::validation::{
    native_integer_value_is_exact_f64, value_has_logical_class, value_has_native_integer_class,
};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_builtins::{
    QqplotInputRole, QQPLOT_EXPLICIT_GPU_EXTENSION, QQPLOT_LOGICAL_INPUT_EXTENSION,
};
use runmat_value::Value;

pub(super) fn admit_host_visible(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.is_empty() {
        return Err(invalid("qqplot: sample input is required"));
    }
    if args.len() > 3 {
        return Err(invalid("qqplot: accepts at most x, y, and pvec inputs"));
    }
    for (index, value) in args.iter().enumerate() {
        let role = QqplotInputRole::at(index).expect("arity checked above");
        if matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QQPLOT_EXPLICIT_GPU_EXTENSION,
                NAME,
            )?;
        }
        if value_has_logical_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QQPLOT_LOGICAL_INPUT_EXTENSION,
                NAME,
            )?;
        }
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(role.integer_extension(), NAME)?;
        }
        validate_role(value, role, args.len())?;
        if !matches!(value, Value::GpuTensor(_))
            && value_has_native_integer_class(value)
            && !native_integer_value_is_exact_f64(value)
        {
            return Err(invalid(format!(
                "qqplot: integer {} values must be exactly representable as double",
                role.integer_label()
            )));
        }
        if role == QqplotInputRole::Percentages {
            validate_host_percentages(value)?;
        }
    }
    Ok(())
}

fn validate_host_percentages(value: &Value) -> BuiltinResult<()> {
    let valid = |number: f64| number.is_finite() && (0.0..=100.0).contains(&number);
    let host_values_valid = match value {
        Value::Num(number) => valid(*number),
        Value::Int(number) => valid(number.to_f64()),
        Value::Bool(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => true,
        Value::Tensor(tensor) => (0..tensor.len()).all(|index| {
            tensor
                .numeric_value_at(index)
                .is_some_and(|number| valid(number.materialize_f64()))
        }),
        _ => return Ok(()), // the role check has already rejected this class
    };
    if host_values_valid {
        Ok(())
    } else {
        Err(invalid(
            "qqplot: pvec values must be finite percentages in the interval [0, 100]",
        ))
    }
}

fn validate_role(value: &Value, role: QqplotInputRole, argument_count: usize) -> BuiltinResult<()> {
    let (shape, len) = match value {
        Value::Tensor(tensor) => (Some(tensor.shape()), Some(tensor.len())),
        Value::LogicalArray(array) => (Some(array.shape.as_slice()), Some(array.len())),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != GpuTensorStorage::Real {
                return Err(invalid(
                    "qqplot: inputs must be real numeric or logical values",
                ));
            }
            let len = handle
                .shape()
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent));
            (
                Some(handle.shape()),
                Some(len.ok_or_else(|| invalid("qqplot: input shape overflows"))?),
            )
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => (None, Some(1)),
        _ => {
            return Err(invalid(
                "qqplot: inputs must be real numeric or logical values",
            ))
        }
    };
    if matches!(role, QqplotInputRole::SampleX | QqplotInputRole::SampleY) {
        if let Some(shape) = shape {
            if shape.len() > 2 && shape.iter().filter(|&&extent| extent > 1).count() > 1 {
                return Err(invalid("qqplot: input must be a vector or 2-D matrix"));
            }
        }
        let empty_placeholder = role == QqplotInputRole::SampleY && argument_count == 3;
        if len == Some(0) && !empty_placeholder {
            return Err(invalid("qqplot: sample input must not be empty"));
        }
    } else if len == Some(0) {
        return Err(invalid("qqplot: pvec must not be empty"));
    }
    Ok(())
}
