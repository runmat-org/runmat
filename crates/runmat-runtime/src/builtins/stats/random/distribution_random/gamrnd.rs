//! Gamma-distribution random samples.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinErrorDescriptor, GAMRND_ERROR_INTERNAL, GAMRND_ERROR_INVALID_ARGUMENT,
    GAMRND_ERROR_TOO_MANY_OUTPUTS, GAMRND_INTEGER_SCALE_EXTENSION, GAMRND_INTEGER_SHAPE_EXTENSION,
    GAMRND_INTEGER_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use super::{normalize_dims, normalize_shape, RandomArgs};
use crate::builtins::common::{gpu_helpers, random, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "gamrnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::distribution_random::gamrnd"
)]
pub(crate) async fn gamrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    let args = parse_args(args).await?;
    validate_parameters(&args.random)?;
    let len = checked_element_count(&args.random.shape)?;
    let data =
        random::generate_gamma_shape_scale(&args.random.first, &args.random.second, len, "gamrnd")
            .map_err(|err| internal(err.message().to_string()))?;
    args.output.finish(data, args.random.shape)
}

struct GamrndArgs {
    random: RandomArgs,
    output: GamrndOutputPlan,
}

struct GamrndOutputPlan {
    single: bool,
    source: Option<GpuTensorHandle>,
}

impl GamrndOutputPlan {
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
            "gamrnd",
        )
        .map_err(|error| internal(error.message()))?;
        Ok(Self { single, source })
    }

    fn host_value(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
        if self.single {
            return Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
                .map(Value::Tensor)
                .map_err(|err| internal(format!("gamrnd: {err}")));
        }
        Tensor::new(data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| internal(format!("gamrnd: {err}")))
    }

    fn finish(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
        let host = self.host_value(data, shape)?;
        let Some(source) = &self.source else {
            return Ok(host);
        };
        let restored = gpu_helpers::restore_class_preserving_value(source, host, "gamrnd")
            .map_err(|error| internal(error.message()))?;
        if runmat_accelerate_api::handle_is_explicit(source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(internal(
                "gamrnd: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        Ok(restored)
    }
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<GamrndArgs> {
    if args.len() < 2 {
        return Err(invalid("gamrnd: expected a and b"));
    }
    ensure_extensions(&args)?;
    ensure_supported_representations(&args)?;
    let output = GamrndOutputPlan::inspect(&args)?;
    let first = value_to_tensor(&args[0]).await?;
    let second = value_to_tensor(&args[1]).await?;
    ensure_exact_integer_boundary(&first, "shape parameter")?;
    ensure_exact_integer_boundary(&second, "scale parameter")?;
    let (first_data, second_data, parameter_shape) =
        tensor::binary_numeric_tensors(&first, &second, "gamrnd", "gamrnd")
            .map_err(|err| invalid(err.message()))?;
    let shape = if args.len() > 2 {
        parse_shape_args(&args[2..]).await?
    } else {
        normalize_shape(parameter_shape.clone())
    };
    if (first_data.len() != 1 || second_data.len() != 1)
        && normalize_shape(parameter_shape) != shape
    {
        return Err(invalid(
            "gamrnd: requested size must match nonscalar parameters",
        ));
    }
    Ok(GamrndArgs {
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
    tensor::value_into_tensor_for("gamrnd", gathered).map_err(invalid)
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(descriptor_error(
            &GAMRND_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

fn validate_parameters(args: &RandomArgs) -> BuiltinResult<()> {
    if args
        .first
        .iter()
        .any(|value| value.is_nan() || *value < 0.0)
    {
        return Err(invalid("gamrnd: shape parameter must be nonnegative"));
    }
    if args
        .second
        .iter()
        .any(|value| value.is_nan() || *value <= 0.0)
    {
        return Err(invalid("gamrnd: scale parameter must be positive"));
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
            return Err(invalid(format!(
                "gamrnd: {role} inputs must be dense real numeric values"
            )));
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
            crate::compatibility::ensure_builtin_extension_enabled(extension, "gamrnd")?;
        }
    }
    if args.iter().skip(2).any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAMRND_INTEGER_SIZE_EXTENSION,
            "gamrnd",
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn ensure_exact_integer_boundary(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .any(|integer| !crate::builtins::common::validation::integer_is_exact_f64(integer))
    {
        return Err(invalid(format!(
            "gamrnd: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for value in rest {
        let parsed = parse_shape_value(value).await?;
        if rest.len() > 1 && parsed.len() != 1 {
            return Err(invalid("gamrnd: separate size arguments must be scalars"));
        }
        dims.extend(parsed);
    }
    Ok(normalize_dims(dims))
}

async fn parse_shape_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| invalid(format!("gamrnd: {err}")))?;
    let tensor = tensor::value_into_tensor_for("gamrnd", gathered)
        .map_err(|err| invalid(format!("gamrnd: {err}")))?;
    if tensor.len() > 1 && !(tensor.shape.len() == 1 || tensor.shape.first() == Some(&1)) {
        return Err(invalid("gamrnd: size vector must be a row vector"));
    }
    (0..tensor.len())
        .map(|index| {
            parse_size_scalar(
                tensor
                    .numeric_value_at(index)
                    .expect("size tensor index must exist"),
            )
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
        invalid("gamrnd: size values must be finite integers in the supported dimension range")
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
            .ok_or_else(|| invalid("gamrnd: requested size exceeds the supported array bounds"))
    })
}

fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    descriptor_error(&GAMRND_ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl std::fmt::Display) -> RuntimeError {
    descriptor_error(&GAMRND_ERROR_INTERNAL, detail)
}

fn descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin("gamrnd");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "gamrnd/tests.rs"]
mod tests;
