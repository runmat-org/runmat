//! Binomial-distribution random samples.

use runmat_builtins::{
    BINORND_CATALOG_ENTRY, BINORND_ERROR_INTERNAL, BINORND_ERROR_INVALID_ARGUMENT,
    BINORND_ERROR_TOO_MANY_OUTPUTS, BINORND_INTEGER_PROBABILITY_EXTENSION,
    BINORND_INTEGER_SIZE_EXTENSION, BINORND_INTEGER_TRIALS_EXTENSION,
    BINORND_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use super::support::{PreparedRandomArgs, RandomArgs, RandomBoundary};
use crate::builtins::common::random;
use crate::BuiltinResult;

const BOUNDARY: RandomBoundary = RandomBoundary::new(
    &BINORND_CATALOG_ENTRY,
    &BINORND_ERROR_INVALID_ARGUMENT,
    &BINORND_ERROR_INTERNAL,
    &BINORND_ERROR_TOO_MANY_OUTPUTS,
);

#[runtime_builtin(
    name = "binornd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::distribution_random::binornd"
)]
pub(crate) async fn binornd_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    BOUNDARY.reject_excess_outputs()?;
    let prepared = parse_args(args.view()).await?;
    validate_parameters(&prepared.random)?;
    let len = BOUNDARY.checked_element_count(&prepared.random.shape)?;
    let mut turn = random::acquire_turn(None, BOUNDARY.name()).await?;
    let data = random::generate_binomial(
        &mut turn,
        &prepared.random.first,
        &prepared.random.second,
        len,
        BOUNDARY.name(),
    )
    .map_err(|error| BOUNDARY.internal(error.message()))?;
    prepared.finish(&BOUNDARY, data)
}

async fn parse_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<PreparedRandomArgs> {
    if args.len() < 2 {
        return Err(BOUNDARY.invalid("expected trial-count and probability parameters"));
    }
    ensure_extensions(args)?;
    ensure_supported_representations(args)?;
    BOUNDARY.prepare(args, "trial-count", "probability").await
}

fn validate_parameters(args: &RandomArgs) -> BuiltinResult<()> {
    if args
        .first
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || value.fract() != 0.0)
    {
        return Err(BOUNDARY.invalid("number of trials must be a positive integer"));
    }
    if args
        .second
        .iter()
        .any(|value| value.is_nan() || !(0.0..=1.0).contains(value))
    {
        return Err(BOUNDARY.invalid("probability must be between zero and one"));
    }
    Ok(())
}

fn ensure_supported_representations(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (index, value) in args.iter().enumerate() {
        let parameter = index < 2;
        let supported = matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
            || parameter && matches!(value, Value::Bool(_) | Value::LogicalArray(_))
            || matches!(value, Value::GpuTensor(_));
        let real_gpu = !matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        let logical_gpu_allowed = !matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_is_logical(handle) && !parameter);
        if !supported || !real_gpu || !logical_gpu_allowed {
            let role = if parameter { "parameter" } else { "size" };
            return Err(
                BOUNDARY.invalid(format!("{role} inputs must be dense real numeric values"))
            );
        }
    }
    Ok(())
}

fn ensure_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    for (value, extension) in args.iter().take(2).zip([
        &BINORND_INTEGER_TRIALS_EXTENSION,
        &BINORND_INTEGER_PROBABILITY_EXTENSION,
    ]) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, BOUNDARY.name())?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BINORND_LOGICAL_INPUT_EXTENSION,
                BOUNDARY.name(),
            )?;
        }
    }
    if args.iter().skip(2).any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BINORND_INTEGER_SIZE_EXTENSION,
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
fn ensure_exact_integer_boundary(
    tensor: &runmat_value::Tensor,
    role: &str,
) -> Result<(), crate::RuntimeError> {
    BOUNDARY.ensure_exact_integer_boundary(tensor, role)
}

#[cfg(test)]
#[path = "binornd/tests.rs"]
mod tests;
