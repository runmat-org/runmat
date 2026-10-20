//! MATLAB-compatible `hilbert` builtin for analytic signal construction.

use std::mem::size_of;

use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderHilbertRequest};
use runmat_builtins::{
    BuiltinErrorDescriptor, HILBERT_EMPTY_LENGTH_EXTENSION, HILBERT_ERROR_ARG_COUNT,
    HILBERT_ERROR_INTERNAL, HILBERT_ERROR_INVALID_INPUT, HILBERT_ERROR_INVALID_LENGTH,
    HILBERT_INTEGER_DATA_EXTENSION, HILBERT_LOGICAL_DATA_EXTENSION,
    HILBERT_LOGICAL_LENGTH_EXTENSION, HILBERT_ND_INPUT_EXTENSION, HILBERT_TYPED_LENGTH_EXTENSION,
    HILBERT_ZERO_LENGTH_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::common::{
    default_dimension, parse_length, tensor_to_complex_tensor, transform_complex_tensor,
    TransformDirection,
};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "hilbert";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::hilbert")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("analytic-signal"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("signal_hilbert")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Computes the analytic signal using FFT-domain one-sided spectrum weighting. Providers can implement `signal_hilbert` for resident real GPU tensors; the runtime gathers to host when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::hilbert")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Hilbert transforms are FFT-domain operations and terminate fusion plans.",
};

fn hilbert_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    hilbert_error_with_message(error.message, error)
}

fn hilbert_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    hilbert_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn hilbert_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hilbert_terminal_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hilbert_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "hilbert",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::hilbert"
)]
async fn hilbert_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    cooperation.completed(0).await?;
    ensure_hilbert_extensions(&value, rest.view())?;
    let length = parse_arguments(rest.view())?;
    let result = match value {
        Value::GpuTensor(handle) => hilbert_gpu_tensor(handle, length, &mut cooperation).await,
        Value::Complex(re, _) => hilbert_tensor(
            Tensor::new(vec![re], vec![1, 1])
                .map_err(|source| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, &source))?,
            length,
        ),
        Value::ComplexTensor(tensor) => hilbert_tensor(real_part_tensor(tensor)?, length),
        other => {
            crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
                &other,
                BUILTIN_NAME,
            )?;
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other).map_err(|detail| {
                hilbert_error_with_detail(&HILBERT_ERROR_INVALID_INPUT, detail)
            })?;
            hilbert_tensor(tensor, length)
        }
    }?;
    if let Err(error) = cooperation.completed(0).await {
        return match result {
            Value::GpuTensor(output) => Err(gpu_helpers::reject_gpu_output(output, error)),
            _ => Err(error),
        };
    }
    Ok(result)
}

async fn hilbert_gpu_tensor(
    handle: GpuTensorHandle,
    length: Option<usize>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let provider = super::resident_output::owner(&handle, || {
        hilbert_terminal_error(
            &HILBERT_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(&handle);
    let complex_input =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    let integer_input = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    let logical_input = runmat_accelerate_api::handle_is_logical(&handle);

    let mut shape = handle.shape().to_vec();
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_one_based = default_dimension(&shape);
    let dim_index = dim_one_based - 1;
    validate_transform_allocation(&shape, dim_index, length)?;

    let current_len = shape.get(dim_index).copied().unwrap_or(1);
    let target_len = length.unwrap_or(current_len);
    if target_len != 0 && !complex_input && !integer_input && !logical_input {
        cooperation.completed(0).await?;
        let request = ProviderHilbertRequest {
            input: &handle,
            length,
            dim: dim_index,
        };
        if provider.supports_signal_hilbert(&request) {
            match provider.signal_hilbert(&request).await {
                Ok(out) => {
                    if let Err(error) = cooperation.completed(0).await {
                        return Err(gpu_helpers::reject_gpu_output(out, error));
                    }
                    let out = super::resident_output::OutputContext {
                        input: &handle,
                        provider,
                        caller_inputs: &[],
                    }
                    .admit(
                        out,
                        &gpu_helpers::BinaryGpuOutputContract {
                            shape: logical_shape_for_hilbert(&shape, dim_index, target_len),
                            storage: GpuTensorStorage::ComplexInterleaved,
                            precision: runmat_accelerate_api::handle_precision(&handle),
                            integer: None,
                            logical: false,
                            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
                        },
                        &[],
                        || {
                            hilbert_terminal_error(
                                &HILBERT_ERROR_INTERNAL,
                                "provider signal_hilbert returned malformed output",
                            )
                        },
                    )?;
                    return gpu_helpers::complex_gpu_value(out);
                }
                Err(err) => {
                    return Err(crate::runtime_error::with_provider_execution_source(
                        hilbert_terminal_error(
                            &HILBERT_ERROR_INTERNAL,
                            format!("provider signal_hilbert failed: {err}"),
                        ),
                        err,
                    ));
                }
            }
        }
    }

    cooperation.completed(0).await?;
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|source| {
            hilbert_error_with_source(&HILBERT_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &gathered,
        BUILTIN_NAME,
    )?;
    let host = match gathered {
        Value::ComplexTensor(tensor) => hilbert_tensor(real_part_tensor(tensor)?, length)?,
        Value::Complex(re, _) => hilbert_tensor(
            Tensor::new(vec![re], shape.clone())
                .map_err(|source| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, &source))?,
            length,
        )?,
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other).map_err(|detail| {
                hilbert_error_with_detail(&HILBERT_ERROR_INVALID_INPUT, detail)
            })?;
            hilbert_tensor(tensor, length)?
        }
    };
    cooperation.completed(0).await?;
    super::resident_output::restore(&handle, host, BUILTIN_NAME, &[])
}

fn ensure_hilbert_extensions(
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let integer_data = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer_data {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HILBERT_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical_data = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical_data {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HILBERT_LOGICAL_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let rank = match value {
        Value::Tensor(tensor) => tensor.shape().len(),
        Value::ComplexTensor(tensor) => tensor.shape().len(),
        Value::LogicalArray(array) => array.shape.len(),
        Value::GpuTensor(handle) => handle.shape().len(),
        _ => 2,
    };
    if rank > 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HILBERT_ND_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.len() == 1 {
        let length = args.first().expect("one Hilbert length argument");
        let typed_integer = matches!(length, Value::Int(_))
            || matches!(length, Value::Tensor(tensor) if tensor.integer_storage().is_some());
        if typed_integer {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HILBERT_TYPED_LENGTH_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(length, Value::Bool(_) | Value::LogicalArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HILBERT_LOGICAL_LENGTH_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if hilbert_length_is_empty(length) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HILBERT_EMPTY_LENGTH_EXTENSION,
                BUILTIN_NAME,
            )?;
        } else if hilbert_length_is_zero(length) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HILBERT_ZERO_LENGTH_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn hilbert_length_is_empty(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.is_empty())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.is_empty())
}

fn hilbert_length_is_zero(value: &Value) -> bool {
    match value {
        Value::Num(value) => *value == 0.0,
        Value::Int(value) => value.try_to_usize() == Some(0),
        Value::Bool(value) => !*value,
        Value::LogicalArray(array) if array.data.len() == 1 => array.data[0] == 0,
        Value::Tensor(tensor) if tensor.len() == 1 => tensor::tensor_value_f64(tensor, 0) == 0.0,
        _ => false,
    }
}

fn real_part_tensor(tensor: ComplexTensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = match tensor.into_complex_storage() {
        ComplexStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(|(real, _)| real).collect())
        }
        ComplexStorage::F32(values) => {
            NumericStorage::F32(values.into_iter().map(|(real, _)| real).collect())
        }
        ComplexStorage::Integer(_) => {
            return Err(hilbert_error_with_detail(
                &HILBERT_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ));
        }
    };
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|source| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, &source))
}

fn logical_shape_for_hilbert(shape: &[usize], dim: usize, length: usize) -> Vec<usize> {
    let mut output = shape.to_vec();
    while output.len() <= dim {
        output.push(1);
    }
    output[dim] = length;
    output
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HilbertGpuMetadata {
    descriptor: runmat_accelerate_api::GpuTensorDescriptor,
    logical: bool,
}

#[cfg(test)]
fn hilbert_input_metadata(handle: &GpuTensorHandle) -> HilbertGpuMetadata {
    HilbertGpuMetadata {
        descriptor: handle.descriptor(),
        logical: runmat_accelerate_api::handle_is_logical(handle),
    }
}

fn parse_arguments(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Option<usize>> {
    match args.len() {
        0 => Ok(None),
        1 => parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
            hilbert_error_with_source(&HILBERT_ERROR_INVALID_LENGTH, "length parse failed", source)
        }),
        _ => Err(hilbert_error(&HILBERT_ERROR_ARG_COUNT)),
    }
}

fn hilbert_tensor(tensor: Tensor, length: Option<usize>) -> BuiltinResult<Value> {
    let complex = tensor_to_complex_tensor(tensor, BUILTIN_NAME).map_err(|source| {
        hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "input promotion failed", source)
    })?;
    let analytic = analytic_signal(complex, length)?;
    Ok(complex_tensor_into_value(analytic))
}

fn analytic_signal(tensor: ComplexTensor, length: Option<usize>) -> BuiltinResult<ComplexTensor> {
    let mut shape = tensor.shape().to_vec();
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_one_based = default_dimension(&shape);
    let dim_index = dim_one_based - 1;
    validate_transform_allocation(&shape, dim_index, length)?;

    let spectrum = transform_complex_tensor(
        tensor,
        length,
        Some(dim_one_based),
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "fft failed", source))?;
    let filtered = apply_analytic_signal_mask(spectrum, dim_index)?;
    transform_complex_tensor(
        filtered,
        None,
        Some(dim_one_based),
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| hilbert_error_with_source(&HILBERT_ERROR_INTERNAL, "ifft failed", source))
}

fn validate_transform_allocation(
    shape: &[usize],
    dim_index: usize,
    length: Option<usize>,
) -> BuiltinResult<()> {
    let mut logical_shape = shape.to_vec();
    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }
    let current_len = logical_shape[dim_index];
    let target_len = length.unwrap_or(current_len);
    if target_len == 0 {
        return Ok(());
    }

    let inner_stride = checked_product(&logical_shape[..dim_index])?;
    let outer_stride = checked_product(&logical_shape[dim_index + 1..])?;
    let num_slices = inner_stride.checked_mul(outer_stride).ok_or_else(|| {
        hilbert_error_with_detail(&HILBERT_ERROR_INVALID_LENGTH, "shape is too large")
    })?;
    let output_len = target_len.checked_mul(num_slices).ok_or_else(|| {
        hilbert_error_with_detail(
            &HILBERT_ERROR_INVALID_LENGTH,
            "requested length is too large",
        )
    })?;
    let max_complex_vec_len = isize::MAX as usize / size_of::<Complex<f64>>();
    if target_len > max_complex_vec_len || output_len > max_complex_vec_len {
        return Err(hilbert_error_with_detail(
            &HILBERT_ERROR_INVALID_LENGTH,
            "requested length is too large",
        ));
    }
    Ok(())
}

fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            hilbert_error_with_detail(&HILBERT_ERROR_INVALID_LENGTH, "shape is too large")
        })
    })
}

fn apply_analytic_signal_mask(
    spectrum: ComplexTensor,
    dim_index: usize,
) -> BuiltinResult<ComplexTensor> {
    let dtype = spectrum.numeric_dtype();
    let mut shape = spectrum.shape().to_vec();
    let mut data = spectrum.materialize_f64();
    while shape.len() <= dim_index {
        shape.push(1);
    }

    let len = shape[dim_index];
    if len == 0 || data.is_empty() {
        return Ok(spectrum);
    }

    let inner_stride = shape[..dim_index]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    let outer_stride = shape[dim_index + 1..]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));

    for outer in 0..outer_stride {
        let base = outer.saturating_mul(len.saturating_mul(inner_stride));
        for inner in 0..inner_stride {
            for freq in 0..len {
                let idx = base + inner + freq * inner_stride;
                let Some(slot) = data.get_mut(idx) else {
                    return Err(hilbert_error_with_detail(
                        &HILBERT_ERROR_INTERNAL,
                        "frequency mask index out of bounds",
                    ));
                };
                let scale = analytic_signal_multiplier(freq, len);
                let value = Complex::new(slot.0, slot.1) * scale;
                *slot = (value.re, value.im);
            }
        }
    }

    let (_, shape) = spectrum.into_complex_storage_and_shape();
    ComplexTensor::from_f64_values_with_dtype(data, shape, dtype)
        .map_err(|error| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, error))
}

fn analytic_signal_multiplier(freq: usize, len: usize) -> f64 {
    if len == 0 {
        return 0.0;
    }
    if freq == 0 {
        return 1.0;
    }
    if len.is_multiple_of(2) {
        if freq < len / 2 {
            2.0
        } else if freq == len / 2 {
            1.0
        } else {
            0.0
        }
    } else if freq <= len / 2 {
        2.0
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "hilbert/tests.rs"]
pub(crate) mod tests;
