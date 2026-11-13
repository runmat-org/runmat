//! MATLAB-compatible `imfilter` builtin implementing multidimensional correlation and
//! convolution with configurable padding strategies.

#[cfg(test)]
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, HostTensorView,
};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ImfilterMode, ImfilterOptions, ImfilterPadding,
    ImfilterShape, IntegerElementType, ProviderPrecision,
};
#[cfg(test)]
use runmat_builtins::IMFILTER_INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinErrorDescriptor, ImfilterOption, IMFILTER_ERROR_INTERNAL, IMFILTER_ERROR_INVALID_INPUT,
    IMFILTER_ERROR_INVALID_OPTION, IMFILTER_FILL_EXTENSION, IMFILTER_VALID_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType};
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(test)]
use super::kernel::{apply_imfilter_tensor, reflect_index};
use super::kernel::{
    build_imfilter_plan, filter_output_shape, floating_accumulation, FilterAccumulation,
};
#[path = "imfilter/host_kernel.rs"]
mod host_kernel;
use host_kernel::{apply_imfilter_host, fallback_accumulation};
#[path = "imfilter/provider.rs"]
mod provider;
use provider::imfilter_gpu;
mod host_route;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::imfilter")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imfilter",
    op_kind: GpuOpKind::Custom("imfilter"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("imfilter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side filtering when available; otherwise gathers to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::imfilter")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imfilter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not a fusion candidate; emits standalone correlation kernels.",
};

const IMFILTER_BUILTIN: &str = "imfilter";

fn filter_error(builtin: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

#[path = "imfilter/kernel_error.rs"]
mod kernel_error;

fn imfilter_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: Option<&str>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => format!("{}: {}", error.message, detail),
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(IMFILTER_BUILTIN);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imfilter_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    imfilter_descriptor_error(error, Some(detail.as_ref()))
}

fn imfilter_map_error(
    err: RuntimeError,
    fallback: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = imfilter_error_with_detail(fallback, err.message());
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "imfilter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::filters::imfilter"
)]
async fn imfilter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    image: Value,
    kernel: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let options = parse_imfilter_options(rest.view())?;
    preflight_imfilter_geometry(&image, &kernel, options.shape)?;
    match (image, kernel) {
        (Value::GpuTensor(image_handle), Value::GpuTensor(filter_handle)) => {
            imfilter_gpu(
                runtime.as_ref(),
                image_handle,
                Value::GpuTensor(filter_handle),
                options,
            )
            .await
        }
        (Value::GpuTensor(image_handle), filter_value) => {
            imfilter_gpu(runtime.as_ref(), image_handle, filter_value, options).await
        }
        (image_value, Value::GpuTensor(filter_handle)) => {
            host_route::run(runtime.as_ref(), image_value, filter_handle, options).await
        }
        (image_value, filter_value) => {
            let filter_tensor = host_kernel_tensor(filter_value)?;
            imfilter_host_value(image_value, filter_tensor, options)
        }
    }
}

fn imfilter_value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(value) => Some(value.shape().to_vec()),
        Value::LogicalArray(value) => Some(value.shape.to_vec()),
        Value::GpuTensor(value) => Some(value.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(vec![1, 1]),
        _ => None,
    }
}

fn preflight_imfilter_geometry(
    image: &Value,
    kernel: &Value,
    shape: ImfilterShape,
) -> BuiltinResult<()> {
    if let Some(kernel_shape) = imfilter_value_shape(kernel) {
        validate_public_kernel_shape(&kernel_shape)?;
        if let Some(image_shape) = imfilter_value_shape(image) {
            filter_output_shape(&image_shape, &kernel_shape, shape, IMFILTER_BUILTIN)
                .map_err(|error| imfilter_map_error(error.into(), &IMFILTER_ERROR_INVALID_INPUT))?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageOutputClass {
    Double,
    Single,
    Integer(NumericDType),
    Logical,
}

impl ImageOutputClass {
    fn from_host_value(value: &Value) -> BuiltinResult<Self> {
        let class = match value {
            Value::Num(_) => Self::Double,
            Value::Int(value) => Self::from_integer_value(value)?,
            Value::Bool(_) | Value::LogicalArray(_) => Self::Logical,
            Value::Tensor(tensor) => Self::from_dtype(tensor.numeric_dtype())?,
            other => {
                return Err(imfilter_error_with_detail(
                    &IMFILTER_ERROR_INVALID_INPUT,
                    format!("A must be a real numeric or logical array, got {other:?}"),
                ))
            }
        };
        Ok(class)
    }

    fn from_integer_value(value: &IntValue) -> BuiltinResult<Self> {
        Self::from_dtype(match value {
            IntValue::I8(_) => NumericDType::I8,
            IntValue::I16(_) => NumericDType::I16,
            IntValue::I32(_) => NumericDType::I32,
            IntValue::I64(_) => NumericDType::I64,
            IntValue::U8(_) => NumericDType::U8,
            IntValue::U16(_) => NumericDType::U16,
            IntValue::U32(_) => NumericDType::U32,
            IntValue::U64(_) => NumericDType::U64,
        })
    }

    fn from_dtype(dtype: NumericDType) -> BuiltinResult<Self> {
        match dtype {
            NumericDType::F64 => Ok(Self::Double),
            NumericDType::F32 => Ok(Self::Single),
            NumericDType::I8
            | NumericDType::I16
            | NumericDType::I32
            | NumericDType::U8
            | NumericDType::U16
            | NumericDType::U32 => Ok(Self::Integer(dtype)),
            NumericDType::I64 | NumericDType::U64 => Err(imfilter_error_with_detail(
                &IMFILTER_ERROR_INVALID_INPUT,
                format!(
                    "A class {} is not supported; expected single, double, int8, int16, int32, uint8, uint16, uint32, or logical",
                    dtype.class_name()
                ),
            )),
        }
    }

    fn from_gpu_handle(handle: &GpuTensorHandle) -> BuiltinResult<Self> {
        if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
            return Err(imfilter_error_with_detail(
                &IMFILTER_ERROR_INVALID_INPUT,
                "A must be real",
            ));
        }
        let class = if runmat_accelerate_api::handle_is_logical(handle) {
            Self::Logical
        } else if let Some(integer) = runmat_accelerate_api::handle_integer_type(handle) {
            Self::from_dtype(match integer {
                IntegerElementType::I8 => NumericDType::I8,
                IntegerElementType::I16 => NumericDType::I16,
                IntegerElementType::I32 => NumericDType::I32,
                IntegerElementType::I64 => NumericDType::I64,
                IntegerElementType::U8 => NumericDType::U8,
                IntegerElementType::U16 => NumericDType::U16,
                IntegerElementType::U32 => NumericDType::U32,
                IntegerElementType::U64 => NumericDType::U64,
            })?
        } else {
            match runmat_accelerate_api::handle_precision(handle) {
                Some(ProviderPrecision::F32) => Self::Single,
                Some(ProviderPrecision::F64) => Self::Double,
                None => {
                    return Err(imfilter_error_with_detail(
                        &IMFILTER_ERROR_INVALID_INPUT,
                        "A GPU class metadata is unavailable",
                    ))
                }
            }
        };
        if !gpu_helpers::gpu_class_metadata_matches(
            handle,
            class.provider_precision(),
            runmat_accelerate_api::handle_integer_type(handle),
            class == Self::Logical,
        ) {
            return Err(imfilter_error_with_detail(
                &IMFILTER_ERROR_INVALID_INPUT,
                "A GPU class metadata contradicts its physical storage",
            ));
        }
        Ok(class)
    }

    fn provider_precision(self) -> Option<ProviderPrecision> {
        match self {
            Self::Double => Some(ProviderPrecision::F64),
            Self::Single => Some(ProviderPrecision::F32),
            Self::Integer(_) | Self::Logical => None,
        }
    }
}

fn host_kernel_tensor(value: Value) -> BuiltinResult<Tensor> {
    let tensor = match value {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, err))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(imfilter_error_with_detail(
                &IMFILTER_ERROR_INVALID_INPUT,
                format!("H must be a double array, got {other:?}"),
            ))
        }
    };
    if tensor.numeric_dtype() != NumericDType::F64 {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            format!(
                "H must be double, got {}",
                tensor.numeric_dtype().class_name()
            ),
        ));
    }
    validate_public_kernel_tensor(&tensor)?;
    Ok(tensor)
}

fn validate_public_kernel_tensor(tensor: &Tensor) -> BuiltinResult<()> {
    validate_public_kernel_shape(tensor.shape())
}

fn validate_public_kernel_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.contains(&0) {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be non-empty",
        ));
    }
    if shape.len() > 2 {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be a double vector or matrix",
        ));
    }
    Ok(())
}

async fn download_kernel_tensor(handle: &GpuTensorHandle) -> BuiltinResult<Tensor> {
    validate_public_kernel_shape(handle.shape())?;
    if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_precision(handle) != Some(ProviderPrecision::F64)
    {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be a real double array",
        ));
    }
    if !gpu_helpers::gpu_class_metadata_matches(handle, Some(ProviderPrecision::F64), None, false) {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H GPU class metadata contradicts double storage",
        ));
    }
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(handle);
    let result = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await;
    let value = result.map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    host_kernel_tensor(value)
}

fn imfilter_host_value(
    image_value: Value,
    kernel_tensor: Tensor,
    options: ImfilterOptions,
) -> BuiltinResult<Value> {
    let output_class = ImageOutputClass::from_host_value(&image_value)?;
    let image_tensor = tensor::value_into_tensor_for(IMFILTER_BUILTIN, image_value)
        .map_err(|err| imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, err))?;
    let result = apply_imfilter_host(
        &image_tensor,
        &kernel_tensor,
        &options,
        output_class,
        FilterAccumulation::HostDouble,
        IMFILTER_BUILTIN,
    )
    .map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    Ok(result)
}

fn parse_imfilter_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ImfilterOptions> {
    let mut options = ImfilterOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let mut consumed = 0usize;
        if matches_numeric_scalar(&args[idx]) {
            let scalar = parse_scalar(IMFILTER_BUILTIN, &args[idx])
                .map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_OPTION))?;
            options.padding = ImfilterPadding::Constant;
            options.constant_value = scalar;
        } else if let Some(text) = tensor::value_to_string(&args[idx]) {
            match ImfilterOption::from_name(&text) {
                Some(ImfilterOption::Replicate) => options.padding = ImfilterPadding::Replicate,
                Some(ImfilterOption::Symmetric) => options.padding = ImfilterPadding::Symmetric,
                Some(ImfilterOption::Circular) => options.padding = ImfilterPadding::Circular,
                Some(ImfilterOption::Fill) => {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &IMFILTER_FILL_EXTENSION,
                        IMFILTER_BUILTIN,
                    )?;
                    options.padding = ImfilterPadding::Constant;
                    if let Some(next) = args.get(idx + 1) {
                        if matches_numeric_scalar(next) {
                            let scalar = parse_scalar(IMFILTER_BUILTIN, next)
                                .map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_OPTION))?;
                            options.constant_value = scalar;
                            consumed = 1;
                        } else if tensor::value_to_string(next).is_some() {
                            options.constant_value = 0.0;
                        } else {
                            return Err(imfilter_error_with_detail(
                                &IMFILTER_ERROR_INVALID_OPTION,
                                "expected numeric pad value after 'fill'",
                            ));
                        }
                    } else {
                        options.constant_value = 0.0;
                    }
                }
                Some(ImfilterOption::Same) => options.shape = ImfilterShape::Same,
                Some(ImfilterOption::Full) => options.shape = ImfilterShape::Full,
                Some(ImfilterOption::Valid) => {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &IMFILTER_VALID_EXTENSION,
                        IMFILTER_BUILTIN,
                    )?;
                    options.shape = ImfilterShape::Valid;
                }
                Some(ImfilterOption::Convolution) => options.mode = ImfilterMode::Convolution,
                Some(ImfilterOption::Correlation) => options.mode = ImfilterMode::Correlation,
                None => {
                    return Err(imfilter_error_with_detail(
                        &IMFILTER_ERROR_INVALID_OPTION,
                        format!(
                            "unknown option '{}' (supported: 'same', 'full', 'valid', 'replicate', 'symmetric', 'circular', 'fill', 'conv', 'corr')",
                            text.trim().to_ascii_lowercase()
                        ),
                    ))
                }
            }
        } else {
            return Err(imfilter_error_with_detail(
                &IMFILTER_ERROR_INVALID_OPTION,
                format!(
                    "unsupported option {:?}; expected string flags or numeric pad values",
                    args[idx]
                ),
            ));
        }
        idx += 1 + consumed;
    }
    Ok(options)
}

fn matches_numeric_scalar(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    )
}

fn parse_scalar(builtin: &str, value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                Ok(tensor::tensor_value_f64(t, 0))
            } else {
                Err(filter_error(
                    builtin,
                    format!(
                        "{builtin}: expected scalar value, got tensor of size {}",
                        tensor_element_len(t)
                    ),
                ))
            }
        }
        Value::LogicalArray(la) => {
            if la.data.len() == 1 {
                Ok(if la.data[0] != 0 { 1.0 } else { 0.0 })
            } else {
                Err(filter_error(
                    builtin,
                    format!(
                        "{builtin}: expected scalar logical value, got array of size {}",
                        la.data.len()
                    ),
                ))
            }
        }
        other => Err(filter_error(
            builtin,
            format!("{builtin}: expected numeric scalar, got {:?}", other),
        )),
    }
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

#[cfg(test)]
#[path = "imfilter/tests.rs"]
pub(crate) mod tests;
