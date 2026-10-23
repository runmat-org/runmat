use super::*;
fn integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_has_native_integer_class(value)
}

pub(super) fn admit(args: &IntegrationArguments<'_>) -> BuiltinResult<()> {
    let extension = |extension: &'static runmat_builtins::BuiltinExtensionDescriptor| {
        crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)
    };
    if integer(args.y) {
        extension(&runmat_builtins::TRAPZ_INTEGER_DATA_EXTENSION)?;
    }
    if args.spacing.is_some_and(integer) {
        extension(&runmat_builtins::TRAPZ_INTEGER_SPACING_EXTENSION)?;
    }
    for value in std::iter::once(args.y)
        .chain(args.spacing)
        .filter(|value| !matches!(value, Value::GpuTensor(_)))
    {
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
            return Err(trapz_error_with_detail(
                &TRAPZ_ERROR_INVALID_INPUT,
                "integer sample data and spacing must be exactly representable as double",
            ));
        }
    }
    Ok(())
}
