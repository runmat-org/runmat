//! Ridge regression compatibility surface.

use runmat_builtins::{
    BuiltinErrorDescriptor, RidgeScaling, RIDGE_ERROR_INTERNAL, RIDGE_ERROR_INVALID_ARGUMENT,
    RIDGE_INTEGER_DATA_EXTENSION, RIDGE_INTEGER_K_EXTENSION, RIDGE_INTEGER_SCALED_EXTENSION,
};
pub use runmat_builtins::{RIDGE_DESCRIPTOR, RIDGE_EXTENSIONS, RIDGE_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "ridge";
const EPS: f64 = 1.0e-12;

fn ridge_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    ridge_error(message, &RIDGE_ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    ridge_error(message, &RIDGE_ERROR_INTERNAL)
}

mod compute;
use compute::ridge_compute;
mod admission;
use admission::{parse_resident_scaled, preflight_data, preflight_scaled, value_to_tensor};
#[runtime_builtin(
    name = "ridge",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::ridge"
)]
async fn ridge_builtin(
    y: Value,
    x: Value,
    k: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "ridge: accepts at most one scaled argument",
        ));
    }
    preflight_data(&y, &x, &k)?;
    let scaled = rest.first().map(preflight_scaled).transpose()?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &RIDGE_INTEGER_DATA_EXTENSION,
        NAME,
        "response",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &RIDGE_INTEGER_DATA_EXTENSION,
        NAME,
        "predictor",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &k,
        &RIDGE_INTEGER_K_EXTENSION,
        NAME,
        "regularization parameter",
    )
    .await?;
    if let Some(value) = rest.first() {
        if crate::builtins::common::validation::value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RIDGE_INTEGER_SCALED_EXTENSION,
                NAME,
            )?;
        }
    }
    let y = value_to_tensor(y).await?;
    let x = value_to_tensor(x).await?;
    let k = value_to_tensor(k).await?;
    let scaled = match scaled {
        Some(Some(scaled)) => scaled,
        Some(None) => parse_resident_scaled(&rest[0]).await?,
        None => RidgeScaling::Standardized,
    };
    ridge_compute(y, x, k, scaled == RidgeScaling::Standardized).map(Value::Tensor)
}

#[cfg(test)]
mod tests;
