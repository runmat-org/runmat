use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(not(feature = "plot-core"))]
use runmat_builtins::CLOSE_ERROR_INVALID_ARGUMENT;
use runmat_builtins::{
    CLOSE_DESCRIPTOR, CLOSE_ERROR_TOO_MANY_OUTPUTS, CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLOSE_VARIADIC_TARGETS_EXTENSION,
};

#[runtime_builtin(
    name = "close",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::close"
)]
pub async fn close_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &CLOSE_DESCRIPTOR,
        &CLOSE_ERROR_TOO_MANY_OUTPUTS,
        "close",
    )?;
    if args.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLOSE_VARIADIC_TARGETS_EXTENSION,
            "close",
        )?;
    }
    if args.iter().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
            "close",
        )?;
    }
    if let Some(status) =
        crate::builtins::io::net::close::close_if_network_targets(runtime.as_ref(), args.view())
            .await?
    {
        return Ok(status);
    }

    close_plotting_targets(args.view())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[cfg(feature = "plot-core")]
fn close_plotting_targets(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<f64> {
    crate::builtins::plotting::close::close_plot_targets(args)
}

#[cfg(not(feature = "plot-core"))]
fn close_plotting_targets(
    _args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<f64> {
    let mut builder =
        crate::build_runtime_error(CLOSE_ERROR_INVALID_ARGUMENT.message).with_builtin("close");
    if let Some(identifier) = CLOSE_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}
