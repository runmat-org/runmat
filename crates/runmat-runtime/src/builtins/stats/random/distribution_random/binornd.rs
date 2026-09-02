//! Binomial-distribution random samples.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinErrorDescriptor, BINORND_ERROR_INTERNAL, BINORND_ERROR_INVALID_ARGUMENT,
    BINORND_ERROR_TOO_MANY_OUTPUTS, BINORND_INTEGER_PROBABILITY_EXTENSION,
    BINORND_INTEGER_SIZE_EXTENSION, BINORND_INTEGER_TRIALS_EXTENSION,
    BINORND_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use super::{normalize_dims, normalize_shape, RandomArgs};
use crate::builtins::common::{gpu_helpers, random, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "binornd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::distribution_random::binornd"
)]
pub(crate) async fn binornd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    let args = parse_args(args).await?;
    validate_parameters(&args.random)?;
    let len = checked_element_count(&args.random.shape)?;
    let data = random::generate_binomial(&args.random.first, &args.random.second, len, "binornd")
        .map_err(|error| internal(error.message()))?;
    args.output.finish(data, args.random.shape)
}

struct BinorndArgs {
    random: RandomArgs,
    output: BinorndOutputPlan,
}

struct BinorndOutputPlan {
    single: bool,
    source: Option<GpuTensorHandle>,
}

impl BinorndOutputPlan {
    fn inspect(args: &[Value]) -> BuiltinResult<Self> {
        let single = args.iter().take(2).any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle)
                    if runmat_accelerate_api::handle_integer_type(handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(handle)
                        && runmat_accelerate_api::handle_storage(handle)
                            == runmat_accelerate_api::GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_precision(handle)
                            == Some(ProviderPrecision::F32))
        });
        let source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "binornd",
        )
        .map_err(|error| internal(error.message()))?;
        Ok(Self { single, source })
    }

    fn host_value(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
        if self.single {
            return Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
                .map(Value::Tensor)
                .map_err(|error| internal(error.to_string()));
        }
        Tensor::new(data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|error| internal(error.to_string()))
    }

    fn finish(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
        let host = self.host_value(data, shape)?;
        let Some(source) = &self.source else {
            return Ok(host);
        };
        let restored = gpu_helpers::restore_class_preserving_value(source, host, "binornd")
            .map_err(|error| internal(error.message()))?;
        if runmat_accelerate_api::handle_is_explicit(source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(internal(
                "binornd: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        Ok(restored)
    }
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<BinorndArgs> {
    if args.len() < 2 {
        return Err(invalid("binornd: expected n and p"));
    }
    ensure_extensions(&args)?;
    ensure_supported_representations(&args)?;
    let output = BinorndOutputPlan::inspect(&args)?;
    let first = value_to_tensor(&args[0]).await?;
    let second = value_to_tensor(&args[1]).await?;
    ensure_exact_integer_boundary(&first, "trial-count")?;
    ensure_exact_integer_boundary(&second, "probability")?;
    let (first_data, second_data, parameter_shape) =
        tensor::binary_numeric_tensors(&first, &second, "binornd", "binornd")
            .map_err(|error| invalid(error.message()))?;
    let shape = if args.len() > 2 {
        parse_shape_args(&args[2..]).await?
    } else {
        normalize_shape(parameter_shape.clone())
    };
    if (first_data.len() != 1 || second_data.len() != 1)
        && normalize_shape(parameter_shape) != shape
    {
        return Err(invalid(
            "binornd: requested size must match nonscalar parameters",
        ));
    }
    Ok(BinorndArgs {
        random: RandomArgs {
            first: first_data,
            second: second_data,
            shape,
        },
        output,
    })
}

async fn value_to_tensor(value: &Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|error| invalid(error.message()))?;
    tensor::value_into_tensor_for("binornd", gathered).map_err(invalid)
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(descriptor_error(
            &BINORND_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

fn validate_parameters(args: &RandomArgs) -> BuiltinResult<()> {
    if args
        .first
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || value.fract() != 0.0)
    {
        return Err(invalid(
            "binornd: number of trials must be a positive integer",
        ));
    }
    if args
        .second
        .iter()
        .any(|value| value.is_nan() || !(0.0..=1.0).contains(value))
    {
        return Err(invalid("binornd: probability must be between zero and one"));
    }
    Ok(())
}

fn ensure_supported_representations(args: &[Value]) -> BuiltinResult<()> {
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
            return Err(invalid(format!(
                "binornd: {role} inputs must be dense real numeric values"
            )));
        }
    }
    Ok(())
}

fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
    for (value, extension) in args.iter().take(2).zip([
        &BINORND_INTEGER_TRIALS_EXTENSION,
        &BINORND_INTEGER_PROBABILITY_EXTENSION,
    ]) {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, "binornd")?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BINORND_LOGICAL_INPUT_EXTENSION,
                "binornd",
            )?;
        }
    }
    if args.iter().skip(2).any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BINORND_INTEGER_SIZE_EXTENSION,
            "binornd",
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

fn ensure_exact_integer_boundary(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .any(|integer| !crate::builtins::common::validation::integer_is_exact_f64(integer))
    {
        return Err(invalid(format!(
            "binornd: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut dimensions = Vec::new();
    for value in rest {
        let parsed = parse_shape_value(value).await?;
        if rest.len() > 1 && parsed.len() != 1 {
            return Err(invalid("binornd: separate size arguments must be scalars"));
        }
        dimensions.extend(parsed);
    }
    Ok(normalize_dims(dimensions))
}

async fn parse_shape_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|error| invalid(error.message()))?;
    let tensor = tensor::value_into_tensor_for("binornd", gathered).map_err(invalid)?;
    if tensor.len() > 1 && !(tensor.shape.len() == 1 || tensor.shape.first() == Some(&1)) {
        return Err(invalid("binornd: size vector must be a row vector"));
    }
    (0..tensor.len())
        .map(|index| {
            let scalar = tensor
                .numeric_value_at(index)
                .ok_or_else(|| internal("binornd: size tensor storage is inconsistent"))?;
            parse_size_scalar(scalar)
        })
        .collect()
}

fn parse_size_scalar(value: NumericScalar) -> BuiltinResult<usize> {
    let dimension = match value {
        NumericScalar::I8(value) => signed_size(i128::from(value)),
        NumericScalar::I16(value) => signed_size(i128::from(value)),
        NumericScalar::I32(value) => signed_size(i128::from(value)),
        NumericScalar::I64(value) => signed_size(i128::from(value)),
        NumericScalar::U8(value) => unsigned_size(u128::from(value)),
        NumericScalar::U16(value) => unsigned_size(u128::from(value)),
        NumericScalar::U32(value) => unsigned_size(u128::from(value)),
        NumericScalar::U64(value) => unsigned_size(u128::from(value)),
        NumericScalar::F32(value) => floating_size(f64::from(value)),
        NumericScalar::F64(value) => floating_size(value),
    };
    dimension.ok_or_else(|| {
        invalid("binornd: size values must be finite integers in the supported dimension range")
    })
}

fn signed_size(value: i128) -> Option<usize> {
    if value <= 0 {
        Some(0)
    } else {
        usize::try_from(value).ok()
    }
}

fn unsigned_size(value: u128) -> Option<usize> {
    usize::try_from(value).ok()
}

fn floating_size(value: f64) -> Option<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if value <= 0.0 {
        return Some(0);
    }
    if value >= usize::MAX as f64 {
        return None;
    }
    Some(value as usize)
}

fn checked_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| invalid("binornd: requested size exceeds the supported array bounds"))
    })
}

fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    descriptor_error(&BINORND_ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    descriptor_error(&BINORND_ERROR_INTERNAL, detail)
}

fn descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin("binornd");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "binornd/tests.rs"]
mod tests;
