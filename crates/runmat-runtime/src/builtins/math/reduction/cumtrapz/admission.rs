use super::*;
fn integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_has_native_integer_class(value)
}
fn logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value,Value::GpuTensor(h) if runmat_accelerate_api::handle_is_logical(h))
}
pub(super) fn admit(args: &IntegrationArguments<'_>) -> BuiltinResult<()> {
    let extension = |extension: &'static runmat_builtins::BuiltinExtensionDescriptor| {
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)
    };
    if integer(args.y) {
        extension(&runmat_builtins::CUMTRAPZ_INTEGER_Y_EXTENSION)?;
    }
    if logical(args.y) {
        extension(&runmat_builtins::CUMTRAPZ_LOGICAL_Y_EXTENSION)?;
    }
    if args.spacing.is_some_and(integer) {
        extension(&runmat_builtins::CUMTRAPZ_INTEGER_X_EXTENSION)?;
    }
    if args.spacing.is_some_and(logical) {
        extension(&runmat_builtins::CUMTRAPZ_LOGICAL_X_EXTENSION)?;
    }
    if args.dim_value.is_some_and(integer) {
        extension(&runmat_builtins::CUMTRAPZ_INTEGER_DIM_EXTENSION)?;
    }
    if tensor_spacing(args) {
        extension(&runmat_builtins::CUMTRAPZ_TENSOR_SPACING_EXTENSION)?;
    }
    for value in std::iter::once(args.y)
        .chain(args.spacing)
        .filter(|value| !matches!(value, Value::GpuTensor(_)))
    {
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
            return Err(cumtrapz_error_with_detail(
                &CUMTRAPZ_ERROR_INVALID_INPUT,
                "integer sample data and spacing must be exactly representable as double",
            ));
        }
    }
    Ok(())
}
fn tensor_spacing(args: &IntegrationArguments<'_>) -> bool {
    fn shape(value: &Value) -> Option<&[usize]> {
        match value {
            Value::Tensor(t) => Some(t.shape()),
            Value::ComplexTensor(t) => Some(t.shape()),
            Value::LogicalArray(a) => Some(&a.shape),
            Value::GpuTensor(h) => Some(h.shape()),
            _ => None,
        }
    }
    let Some(x) = args.spacing.and_then(shape) else {
        return false;
    };
    let Some(y) = shape(args.y) else { return false };
    if x.iter().filter(|n| **n > 1).count() <= 1 {
        return false;
    }
    let x_len = x.iter().rposition(|n| *n != 1).map_or(0, |i| i + 1);
    let y_len = y.iter().rposition(|n| *n != 1).map_or(0, |i| i + 1);
    x[..x_len] == y[..y_len]
}
