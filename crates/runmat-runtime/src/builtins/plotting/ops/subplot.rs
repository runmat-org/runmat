//! MATLAB-compatible `subplot` builtin for selecting axes within a figure.

use runmat_builtins::{
    BuiltinErrorDescriptor, SUBPLOT_DESCRIPTOR, SUBPLOT_ERROR_INDEX_OUT_OF_RANGE,
    SUBPLOT_ERROR_INTERNAL, SUBPLOT_ERROR_INVALID_ARGUMENT, SUBPLOT_ERROR_TOO_MANY_OUTPUTS,
    SUBPLOT_INTEGER_GRID_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::op_common::cmd_parsing::scalar_from_value;
use super::state::{encode_axes_handle, FigureError};
use crate::{build_runtime_error, RuntimeError};

const BUILTIN_NAME: &str = "subplot";

fn subplot_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    subplot_error_with_message(error.message, error)
}

fn subplot_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_subplot_figure_error(err: FigureError) -> RuntimeError {
    match err {
        FigureError::InvalidSubplotIndex { .. } => subplot_error(&SUBPLOT_ERROR_INDEX_OUT_OF_RANGE),
        FigureError::InvalidSubplotGrid { .. } => subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT),
        other => {
            let mut error = subplot_error_with_message(
                format!("{}: {}", SUBPLOT_ERROR_INTERNAL.message, other),
                &SUBPLOT_ERROR_INTERNAL,
            );
            error.source = Some(Box::new(other));
            error
        }
    }
}

#[runtime_builtin(
    name = "subplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::subplot"
)]
pub fn subplot_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    rows: Value,
    cols: Value,
    position: Value,
) -> crate::BuiltinResult<f64> {
    subplot_with_runtime(runtime.as_ref(), rows, cols, position)
}

#[cfg(test)]
pub fn subplot_builtin(rows: Value, cols: Value, position: Value) -> crate::BuiltinResult<f64> {
    subplot_with_runtime(None, rows, cols, position)
}

fn subplot_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    rows: Value,
    cols: Value,
    position: Value,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &SUBPLOT_DESCRIPTOR,
        &SUBPLOT_ERROR_TOO_MANY_OUTPUTS,
        "subplot",
    )?;
    if [&rows, &cols, &position]
        .into_iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SUBPLOT_INTEGER_GRID_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let m = scalar_from_value(&rows, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let n = scalar_from_value(&cols, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let p = scalar_from_value(&position, BUILTIN_NAME)
        .map_err(|_| subplot_error(&SUBPLOT_ERROR_INVALID_ARGUMENT))?;
    let zero_based = p.saturating_sub(1);
    let property = super::state::PropertyMutationTarget::alias().with_runtime(runtime);
    let axes = super::state::configure_subplot_state_for_property(&property, m, n, zero_based)
        .map_err(map_subplot_figure_error)?;
    Ok(encode_axes_handle(axes.handle, axes.active_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::IntValue;

    #[test]
    fn subplot_returns_axes_handle() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        assert!(matches!(props, Value::Struct(_)));
    }

    #[test]
    fn subplot_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SUBPLOT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"ax = subplot(rows, cols, position)"));
    }

    #[test]
    fn subplot_typed_integer_grid_controls_are_explicitly_gated() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = subplot_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(2)),
            Value::Int(IntValue::I32(2)),
        )
        .expect_err("strict mode rejects typed integer grid controls");
        assert_eq!(
            error.identifier(),
            SUBPLOT_INTEGER_GRID_EXTENSION.error_identifier
        );
        drop(strict);

        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = subplot_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(2)),
            Value::Int(IntValue::I32(2)),
        )
        .expect("extension mode accepts typed integer grid controls");
        assert!(handle.is_finite());
    }
}
