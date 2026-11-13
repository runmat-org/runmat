//! MATLAB-compatible sample-rate conversion builtins.

use runmat_builtins::{
    BuiltinErrorDescriptor, DOWNSAMPLE_INTEGER_FACTOR_EXTENSION,
    DOWNSAMPLE_INTEGER_PHASE_EXTENSION, DOWNSAMPLE_ND_INPUT_EXTENSION, RESAMPLE_ERROR_OUTPUT_COUNT,
    RESAMPLE_INTEGER_OPTION_EXTENSION, SAMPLE_ERROR_ARG_COUNT, SAMPLE_ERROR_BUILD_OUTPUT,
    SAMPLE_ERROR_GATHER_FAILED, SAMPLE_ERROR_INVALID_FACTOR, SAMPLE_ERROR_INVALID_INPUT,
    SAMPLE_ERROR_INVALID_OPTION, SAMPLE_ERROR_INVALID_PHASE, SAMPLE_ERROR_SIZE_OVERFLOW,
    UPSAMPLE_INTEGER_FACTOR_EXTENSION, UPSAMPLE_INTEGER_PHASE_EXTENSION,
    UPSAMPLE_ND_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntegerStorage, NumericStorage, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::{SignalCooperation, SIGNAL_WORK_CHUNK};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod controls;
mod numerics;
mod provider;
use controls::*;
use numerics::*;
use provider::*;

const UPSAMPLE_NAME: &str = "upsample";
const DOWNSAMPLE_NAME: &str = "downsample";
const RESAMPLE_NAME: &str = "resample";
const DEFAULT_RESAMPLE_N: usize = 10;
const DEFAULT_RESAMPLE_BETA: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleOp {
    Up,
    Down,
}

enum SampleInput {
    Real {
        storage: NumericStorage,
        shape: Vec<usize>,
    },
    Complex {
        data: Vec<(f64, f64)>,
        shape: Vec<usize>,
        dtype: runmat_value::NumericDType,
    },
}

impl SampleInput {
    async fn from_value(value: Value, builtin: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                    .await
                    .map_err(|flow| {
                        let detail = flow.message().to_owned();
                        sample_error_with_source(
                            builtin,
                            &SAMPLE_ERROR_GATHER_FAILED,
                            detail,
                            map_control_flow_with_builtin(flow, builtin),
                        )
                    })?;
                match gathered {
                    Value::Tensor(tensor) => Self::from_tensor(tensor, builtin),
                    Value::LogicalArray(logical) => {
                        let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                            sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err)
                        })?;
                        Self::from_tensor(tensor, builtin)
                    }
                    Value::ComplexTensor(tensor) => Self::from_complex_tensor(tensor, builtin),
                    other => Err(sample_error_with_detail(
                        builtin,
                        &SAMPLE_ERROR_INVALID_INPUT,
                        format!("gathered GPU value produced {other:?}"),
                    )),
                }
            }
            Value::Tensor(tensor) => Self::from_tensor(tensor, builtin),
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical).map_err(|err| {
                    sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err)
                })?;
                Self::from_tensor(tensor, builtin)
            }
            Value::ComplexTensor(tensor) => Self::from_complex_tensor(tensor, builtin),
            Value::Num(value) => Ok(Self::Real {
                storage: NumericStorage::F64(vec![value]),
                shape: vec![1, 1],
            }),
            Value::Int(value) => Ok(Self::Real {
                storage: NumericStorage::from_integer_storage(IntegerStorage::from_scalar(value)),
                shape: vec![1, 1],
            }),
            Value::Bool(value) => Ok(Self::Real {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                shape: vec![1, 1],
            }),
            Value::Complex(re, im) => Ok(Self::Complex {
                data: vec![(re, im)],
                shape: vec![1, 1],
                dtype: runmat_value::NumericDType::F64,
            }),
            other => Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_INPUT,
                format!("received {other:?}"),
            )),
        }
    }

    fn from_tensor(tensor: Tensor, builtin: &'static str) -> BuiltinResult<Self> {
        let shape = tensor.shape().to_vec();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| sample_error_with_detail(builtin, &SAMPLE_ERROR_INVALID_INPUT, err))?;
        Ok(Self::Real { storage, shape })
    }

    fn from_complex_tensor(tensor: ComplexTensor, builtin: &'static str) -> BuiltinResult<Self> {
        if tensor.integer_storage().is_some() {
            return Err(sample_error_with_detail(
                builtin,
                &SAMPLE_ERROR_INVALID_INPUT,
                "complex integer samples require an exact integer-complex kernel",
            ));
        }
        Ok(Self::Complex {
            data: tensor.materialize_f64(),
            shape: tensor.shape().to_vec(),
            dtype: tensor.numeric_dtype(),
        })
    }
}

fn sample_error(builtin: &'static str, error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sample_error_with_message(builtin, error.message, error)
}

fn sample_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    sample_error_with_message(
        builtin,
        format!("{}: {}", error.message, detail.as_ref()),
        error,
    )
}

fn sample_error_with_message(
    builtin: &'static str,
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sample_error_with_source(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(builtin)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "upsample",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn upsample_builtin_registered(
    runtime: Option<RuntimeContext>,
    x: Value,
    n: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    upsample_in_runtime(runtime.as_ref(), x, n, rest).await
}

#[cfg(test)]
async fn upsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    upsample_in_runtime(
        None,
        x,
        n,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn upsample_in_runtime(
    runtime: Option<&RuntimeContext>,
    x: Value,
    n: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, UPSAMPLE_NAME)
        .with_integer_error_descriptor(&SAMPLE_ERROR_BUILD_OUTPUT);
    cooperation.completed(0).await?;
    if is_typed_integer_value(&n) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UPSAMPLE_INTEGER_FACTOR_EXTENSION,
            UPSAMPLE_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UPSAMPLE_INTEGER_PHASE_EXTENSION,
            UPSAMPLE_NAME,
        )?;
    }
    if value_rank(&x).is_some_and(|rank| rank > 2) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &UPSAMPLE_ND_INPUT_EXTENSION,
            UPSAMPLE_NAME,
        )?;
    }
    sample_rate_builtin(UPSAMPLE_NAME, SampleOp::Up, x, n, rest, &mut cooperation).await
}

#[runtime_builtin(
    name = "downsample",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn downsample_builtin_registered(
    runtime: Option<RuntimeContext>,
    x: Value,
    n: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    downsample_in_runtime(runtime.as_ref(), x, n, rest).await
}

#[cfg(test)]
async fn downsample_builtin(x: Value, n: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    downsample_in_runtime(
        None,
        x,
        n,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn downsample_in_runtime(
    runtime: Option<&RuntimeContext>,
    x: Value,
    n: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, DOWNSAMPLE_NAME)
        .with_integer_error_descriptor(&SAMPLE_ERROR_BUILD_OUTPUT);
    cooperation.completed(0).await?;
    if is_typed_integer_value(&n) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_INTEGER_FACTOR_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_INTEGER_PHASE_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    if value_rank(&x).is_some_and(|rank| rank > 2) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DOWNSAMPLE_ND_INPUT_EXTENSION,
            DOWNSAMPLE_NAME,
        )?;
    }
    sample_rate_builtin(
        DOWNSAMPLE_NAME,
        SampleOp::Down,
        x,
        n,
        rest,
        &mut cooperation,
    )
    .await
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn value_rank(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape().len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::ComplexTensor(tensor) => Some(tensor.shape().len()),
        Value::GpuTensor(handle) => Some(handle.shape().len()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(2),
        _ => None,
    }
}

/// No-result calls still admit their arguments, but do not gather the signal
/// or allocate a numerical result. Provider identity and physical class are
/// checked from the live handle's authoritative metadata.
fn validate_discarded_resample_input(value: &Value) -> BuiltinResult<()> {
    let admitted = match value {
        Value::Num(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::LogicalArray(logical) => checked_product(&logical.shape) == Some(logical.len()),
        Value::Tensor(tensor) => matches!(
            tensor.numeric_dtype(),
            runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
        ),
        Value::ComplexTensor(tensor) => matches!(
            tensor.numeric_dtype(),
            runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
        ),
        Value::GpuTensor(handle) => {
            let numeric = runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_precision(handle).is_some();
            if numeric && resolved_actual_downsample_owner(handle).is_none() {
                return Err(sample_error_with_detail(
                    RESAMPLE_NAME,
                    &SAMPLE_ERROR_GATHER_FAILED,
                    "resident input has no valid live provider owner or numeric descriptor",
                ));
            }
            numeric
        }
        _ => false,
    };
    if admitted {
        Ok(())
    } else {
        Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_INPUT,
            "resample input must be single, double, or logical",
        ))
    }
}

#[runtime_builtin(
    name = "resample",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::sample_rate"
)]
async fn resample_builtin_registered(
    runtime: Option<RuntimeContext>,
    x: Value,
    p: Value,
    q: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    resample_in_runtime(runtime.as_ref(), x, p, q, rest).await
}

#[cfg(test)]
async fn resample_builtin(x: Value, p: Value, q: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    resample_in_runtime(
        None,
        x,
        p,
        q,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn resample_in_runtime(
    runtime: Option<&RuntimeContext>,
    x: Value,
    p: Value,
    q: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, RESAMPLE_NAME)
        .with_integer_error_descriptor(&SAMPLE_ERROR_BUILD_OUTPUT);
    cooperation.completed(0).await?;
    let requested_outputs = crate::output_count::current_output_count();
    match requested_outputs {
        Some(count) if count > 2 => {
            return Err(sample_error(RESAMPLE_NAME, &RESAMPLE_ERROR_OUTPUT_COUNT));
        }
        _ => {}
    }

    if is_typed_integer_value(&x)
        || matches!(&x, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
    {
        return Err(sample_error_with_detail(
            RESAMPLE_NAME,
            &SAMPLE_ERROR_INVALID_INPUT,
            "resample input must be single or double",
        ));
    }
    let typed_options = is_typed_integer_value(&p)
        || is_typed_integer_value(&q)
        || rest.iter().any(is_typed_integer_value);
    if typed_options {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESAMPLE_INTEGER_OPTION_EXTENSION,
            RESAMPLE_NAME,
        )?;
    }
    let mut p = parse_factor(&p, RESAMPLE_NAME).await?;
    let mut q = parse_factor(&q, RESAMPLE_NAME).await?;
    let divisor = gcd(p, q);
    p /= divisor;
    q /= divisor;
    if requested_outputs == Some(0) {
        validate_resample_options(rest, p, q).await?;
        validate_discarded_resample_input(&x)?;
        return Ok(Value::output_list(Vec::new()));
    }
    let options = parse_resample_options(rest, p, q).await?;
    if let Value::GpuTensor(handle) = &x {
        if let Some(eval) = resample_gpu(handle, p, q, &options).await? {
            return match requested_outputs {
                Some(1) => Ok(Value::output_list(vec![eval.y])),
                Some(2) => Ok(Value::output_list(vec![
                    eval.y,
                    Value::Tensor(filter_tensor(eval.filter.host_values().await?)?),
                ])),
                None => Ok(eval.y),
                _ => unreachable!("output count was validated before evaluation"),
            };
        }
    }
    let input = SampleInput::from_value(x, RESAMPLE_NAME).await?;
    let host_options = HostResampleOptions {
        filter: options.filter.host_values().await?,
        dim: options.dim,
    };
    let eval = apply_resample(input, p, q, &host_options, &mut cooperation).await?;
    match requested_outputs {
        Some(1) => Ok(Value::output_list(vec![eval.y])),
        Some(2) => Ok(Value::output_list(vec![
            eval.y,
            Value::Tensor(filter_tensor(eval.filter.host_values().await?)?),
        ])),
        None => Ok(eval.y),
        _ => unreachable!("output count was validated before evaluation"),
    }
}

async fn sample_rate_builtin(
    builtin: &'static str,
    op: SampleOp,
    x: Value,
    n: Value,
    rest: crate::call::BuiltinArguments,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(sample_error(builtin, &SAMPLE_ERROR_ARG_COUNT));
    }
    let factor = parse_factor(&n, builtin).await?;
    let phase = match rest.first() {
        Some(value) => parse_phase(value, factor, builtin).await?,
        None => 0,
    };
    if let Value::GpuTensor(handle) = &x {
        match op {
            SampleOp::Up => {
                if let Some(output) = upsample_gpu(handle, factor, phase, builtin)? {
                    return Ok(output);
                }
            }
            SampleOp::Down => {
                if let Some(output) = downsample_gpu(handle, factor, phase, builtin)? {
                    return Ok(output);
                }
            }
        }
    }
    let input = SampleInput::from_value(x, builtin).await?;
    apply_sample_rate(input, factor, phase, op, builtin, cooperation).await
}

#[derive(Clone, Debug)]
struct ResampleOptions {
    filter: ResampleFilter,
    dim: Option<usize>,
}

#[derive(Clone, Debug)]
enum ResampleFilter {
    Host(Vec<f64>),
    Gpu {
        handle: runmat_accelerate_api::GpuTensorHandle,
        len: usize,
    },
}

impl ResampleFilter {
    fn len(&self) -> usize {
        match self {
            Self::Host(filter) => filter.len(),
            Self::Gpu { len, .. } => *len,
        }
    }

    fn same_provider_handle(
        &self,
        source: &runmat_accelerate_api::GpuTensorHandle,
        provider: &dyn runmat_accelerate_api::AccelProvider,
    ) -> Option<&runmat_accelerate_api::GpuTensorHandle> {
        match self {
            Self::Gpu { handle, .. }
                if handle.device_id() == source.device_id()
                    && resolved_actual_downsample_owner(handle).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, provider)
                    }) =>
            {
                Some(handle)
            }
            _ => None,
        }
    }

    async fn host_values(&self) -> BuiltinResult<Vec<f64>> {
        match self {
            Self::Host(filter) => Ok(filter.clone()),
            Self::Gpu { handle, .. } => {
                let tensor = gpu_helpers::gather_tensor_async(handle)
                    .await
                    .map_err(|err| {
                        let detail = err.message().to_owned();
                        sample_error_with_source(
                            RESAMPLE_NAME,
                            &SAMPLE_ERROR_GATHER_FAILED,
                            detail,
                            err,
                        )
                    })?;
                Ok(tensor::tensor_into_values_f64(tensor))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct HostResampleOptions {
    filter: Vec<f64>,
    dim: Option<usize>,
}

struct ResampleEval {
    y: Value,
    filter: ResampleFilter,
}

#[cfg(test)]
mod tests;
