use super::*;

pub(super) fn reject_resident_persistence_arguments(
    values: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if values.iter().any(|value| {
        crate::builtins::common::validation::value_has_native_integer_class(value)
            || matches!(value, Value::GpuTensor(_))
    }) {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!(
                "{builtin}: typed integer and gpuArray values are not valid figure, filename, or format arguments"
            ),
        ));
    }
    Ok(())
}

pub(super) fn validate_status_output(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match requested_outputs {
        Some(count) if count > 1 => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected at most one RunMat status output"),
        )),
        Some(1) => {
            if let Some(runtime) = runtime {
                crate::compatibility::ensure_builtin_extension_enabled_in_context(
                    runtime,
                    &runmat_builtins::FIGURE_PERSISTENCE_STATUS_OUTPUT_EXTENSION,
                    builtin,
                )
            } else {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &runmat_builtins::FIGURE_PERSISTENCE_STATUS_OUTPUT_EXTENSION,
                    builtin,
                )
            }
        }
        _ => Ok(()),
    }
}

pub(super) fn validate_load_outputs(
    requested_outputs: Option<usize>,
    maximum: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if requested_outputs.is_some_and(|count| count > maximum) {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: expected at most {maximum} output values"),
        ));
    }
    Ok(())
}
