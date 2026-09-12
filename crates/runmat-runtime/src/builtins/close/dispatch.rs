use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[cfg(not(feature = "plot-core"))]
use super::contract::CLOSE_ERROR_INVALID_ARGUMENT;
use super::contract::{CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION, CLOSE_VARIADIC_TARGETS_EXTENSION};

#[runtime_builtin(
    name = "close",
    category = "general",
    summary = "Close figures or networking resources.",
    keywords = "close,figure,tcpclient,tcpserver,networking",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::close_type),
    descriptor(crate::builtins::close::CLOSE_DESCRIPTOR),
    extensions(crate::builtins::close::CLOSE_EXTENSIONS),
    integer_capabilities(crate::builtins::close::CLOSE_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::close"
)]
pub async fn close_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
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
    if let Some(status) = crate::builtins::io::net::close::close_if_network_targets(&args).await? {
        return Ok(status);
    }

    close_plotting_targets(&args)
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

#[cfg(feature = "plot-core")]
fn close_plotting_targets(args: &[Value]) -> crate::BuiltinResult<f64> {
    crate::builtins::plotting::close::close_plot_targets(args)
}

#[cfg(not(feature = "plot-core"))]
fn close_plotting_targets(_args: &[Value]) -> crate::BuiltinResult<f64> {
    let mut builder =
        crate::build_runtime_error(CLOSE_ERROR_INVALID_ARGUMENT.message).with_builtin("close");
    if let Some(identifier) = CLOSE_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    Err(builder.build())
}
