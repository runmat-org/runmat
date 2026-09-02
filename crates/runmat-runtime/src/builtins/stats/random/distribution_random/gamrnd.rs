//! Gamma-distribution random samples.

use runmat_builtins::{
    GAMRND_CATALOG_ENTRY, GAMRND_ERROR_INTERNAL, GAMRND_ERROR_INVALID_ARGUMENT,
    GAMRND_ERROR_TOO_MANY_OUTPUTS, GAMRND_INTEGER_SCALE_EXTENSION, GAMRND_INTEGER_SHAPE_EXTENSION,
    GAMRND_INTEGER_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::support::{PreparedRandomArgs, RandomArgs, RandomBoundary};
use crate::builtins::common::random;
use crate::BuiltinResult;

const BOUNDARY: RandomBoundary = RandomBoundary::new(
    &GAMRND_CATALOG_ENTRY,
    &GAMRND_ERROR_INVALID_ARGUMENT,
    &GAMRND_ERROR_INTERNAL,
    &GAMRND_ERROR_TOO_MANY_OUTPUTS,
);

#[runtime_builtin(
    name = "gamrnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::distribution_random::gamrnd"
)]
pub(crate) async fn gamrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    BOUNDARY.reject_excess_outputs()?;
    let prepared = parse_args(&args).await?;
    validate_parameters(&prepared.random)?;
    let len = BOUNDARY.checked_element_count(&prepared.random.shape)?;
    let data = random::generate_gamma_shape_scale(
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
        return Err(BOUNDARY.invalid("expected shape and scale parameters"));
    }
    ensure_extensions(args)?;
    ensure_supported_representations(args)?;
    BOUNDARY
        .prepare(args, "shape parameter", "scale parameter")
        .await
}

fn validate_parameters(args: &RandomArgs) -> BuiltinResult<()> {
    if args
        .first
        .iter()
        .any(|value| value.is_nan() || *value < 0.0)
    {
        return Err(BOUNDARY.invalid("shape parameter must be nonnegative"));
    }
    if args
        .second
        .iter()
        .any(|value| value.is_nan() || *value <= 0.0)
    {
        return Err(BOUNDARY.invalid("scale parameter must be positive"));
    }
    Ok(())
}

fn ensure_supported_representations(args: &[Value]) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let supported = matches!(
            value,
            Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
        );
        let real_gpu = !matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                || runmat_accelerate_api::handle_is_logical(handle));
        if !supported || !real_gpu {
            let role = if index < 2 { "parameter" } else { "size" };
            return Err(
                BOUNDARY.invalid(format!("{role} inputs must be dense real numeric values"))
            );
        }
    }
    Ok(())
}

fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
    for (value, extension) in args.iter().take(2).zip([
        &GAMRND_INTEGER_SHAPE_EXTENSION,
        &GAMRND_INTEGER_SCALE_EXTENSION,
    ]) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, BOUNDARY.name())?;
        }
    }
    if args.iter().skip(2).any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAMRND_INTEGER_SIZE_EXTENSION,
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

#[cfg(test)]
fn ensure_exact_integer_boundary(
    tensor: &runmat_value::Tensor,
    role: &str,
) -> Result<(), crate::RuntimeError> {
    BOUNDARY.ensure_exact_integer_boundary(tensor, role)
}

#[cfg(test)]
#[path = "gamrnd/tests.rs"]
mod tests;
