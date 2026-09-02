//! Weibull-distribution random samples.

use runmat_builtins::{
    WBLRND_CATALOG_ENTRY, WBLRND_ERROR_INTERNAL, WBLRND_ERROR_INVALID_ARGUMENT,
    WBLRND_ERROR_TOO_MANY_OUTPUTS, WBLRND_INTEGER_SCALE_EXTENSION, WBLRND_INTEGER_SHAPE_EXTENSION,
    WBLRND_INTEGER_SIZE_EXTENSION, WBLRND_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::support::{PreparedRandomArgs, RandomArgs, RandomBoundary};
use crate::builtins::common::random;
use crate::BuiltinResult;

const BOUNDARY: RandomBoundary = RandomBoundary::new(
    &WBLRND_CATALOG_ENTRY,
    &WBLRND_ERROR_INVALID_ARGUMENT,
    &WBLRND_ERROR_INTERNAL,
    &WBLRND_ERROR_TOO_MANY_OUTPUTS,
);

#[runtime_builtin(
    name = "wblrnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::distribution_random::wblrnd"
)]
pub(crate) async fn wblrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    BOUNDARY.reject_excess_outputs()?;
    let prepared = parse_args(&args).await?;
    validate_parameters(&prepared.random)?;
    let len = BOUNDARY.checked_element_count(&prepared.random.shape)?;
    let data = random::generate_weibull(
        &prepared.random.first,
        &prepared.random.second,
        len,
        BOUNDARY.name(),
    )
    .map_err(|error| BOUNDARY.internal(error.message()))?;
    prepared.finish(&BOUNDARY, data)
}

async fn parse_args(args: &[Value]) -> BuiltinResult<PreparedRandomArgs> {
    if args.len() < 2 {
        return Err(BOUNDARY.invalid("expected scale and shape parameters"));
    }
    ensure_extensions(args)?;
    ensure_supported_representations(args)?;
    BOUNDARY
        .prepare(args, "scale parameter", "shape parameter")
        .await
}

fn validate_parameters(args: &RandomArgs) -> BuiltinResult<()> {
    if args
        .first
        .iter()
        .any(|value| value.is_nan() || *value <= 0.0)
    {
        return Err(BOUNDARY.invalid("scale parameter must be positive"));
    }
    if args
        .second
        .iter()
        .any(|value| value.is_nan() || *value <= 0.0)
    {
        return Err(BOUNDARY.invalid("shape parameter must be positive"));
    }
    Ok(())
}

fn ensure_supported_representations(args: &[Value]) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let supported = matches!(
            value,
            Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
                | Value::Tensor(_)
                | Value::LogicalArray(_)
                | Value::GpuTensor(_)
        );
        let real_gpu = !matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        if !supported || !real_gpu {
            let role = if index < 2 { "parameter" } else { "size" };
            return Err(BOUNDARY.invalid(format!(
                "{role} inputs must be dense real numeric or logical values"
            )));
        }
    }
    Ok(())
}

fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
    for (value, extension) in args.iter().take(2).zip([
        &WBLRND_INTEGER_SCALE_EXTENSION,
        &WBLRND_INTEGER_SHAPE_EXTENSION,
    ]) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, BOUNDARY.name())?;
        }
    }
    if args.iter().skip(2).any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WBLRND_INTEGER_SIZE_EXTENSION,
            BOUNDARY.name(),
        )?;
    }
    if args.iter().any(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WBLRND_LOGICAL_INPUT_EXTENSION,
            BOUNDARY.name(),
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

#[cfg(test)]
#[path = "wblrnd/tests.rs"]
mod tests;
