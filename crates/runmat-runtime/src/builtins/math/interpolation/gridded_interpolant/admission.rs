use super::*;

pub(super) async fn gather_values(
    values: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_preserving_roots_async(values.len(), runtime, |index| {
            crate::dispatcher::gather_if_needed_async(&values[index])
        })
        .await
}

pub(super) fn ensure_constructor_integer_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut numeric_len = args.len();
    let mut methods = 0usize;
    while numeric_len > 0 && methods < 2 {
        let Some(name) = text_from_value(&args[numeric_len - 1])? else {
            break;
        };
        if MethodKeyword::parse(&name).is_none() {
            break;
        }
        numeric_len -= 1;
        methods += 1;
    }
    match numeric_len {
        0 => Ok(()),
        1 => ensure_integer_extension(&args[0], &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values"),
        2 => {
            ensure_integer_extension(&args[0], &GRIDDED_INTEGER_GRID_EXTENSION, "sample grid")?;
            ensure_integer_extension(&args[1], &GRIDDED_INTEGER_VALUES_EXTENSION, "sample values")
        }
        _ => {
            for grid in args.range(..numeric_len - 1)? {
                ensure_integer_extension(grid, &GRIDDED_INTEGER_GRID_EXTENSION, "sample grid")?;
            }
            ensure_integer_extension(
                &args[numeric_len - 1],
                &GRIDDED_INTEGER_VALUES_EXTENSION,
                "sample values",
            )
        }
    }
}

pub(super) fn ensure_query_integer_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for query in args {
        ensure_integer_extension(query, &GRIDDED_INTEGER_QUERY_EXTENSION, "query coordinates")?;
    }
    Ok(())
}

pub(super) fn ensure_integer_extension(
    value: &Value,
    extension: &BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if value_contains_typed_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
        ensure_exact_integer_value(value, role)?;
    }
    Ok(())
}

pub(super) fn value_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        Value::Cell(cell) => cell.data().iter().any(value_contains_typed_integer),
        _ => false,
    }
}

pub(super) fn ensure_exact_integer_value(value: &Value, role: &str) -> BuiltinResult<()> {
    let exact = crate::builtins::common::validation::integer_is_exact_f64;
    let valid = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            (0..storage.len()).all(|index| storage.value_at(index).as_ref().is_some_and(exact))
        }),
        Value::Cell(cell) => cell
            .data()
            .iter()
            .all(|value| ensure_exact_integer_value(value, role).is_ok()),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(format!(
            "integer {role} must be exactly representable as double"
        )))
    }
}
