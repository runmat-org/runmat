//! MATLAB-compatible `imfilter` builtin implementing multidimensional correlation and
//! convolution with configurable padding strategies.

use runmat_accelerate_api::{
    GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, HostTensorView, ImfilterMode,
    ImfilterOptions, ImfilterPadding, ImfilterShape, IntegerElementType, ProviderPrecision,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, LogicalArray, NumericDType, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::image::filters::type_resolvers::imfilter_type;
use crate::builtins::math::fft::common::{gpu_metadata_snapshot, restore_gpu_metadata};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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

const IMFILTER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered image output.",
}];

const IMFILTER_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
];

const IMFILTER_INPUTS_WITH_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input image/array.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter kernel.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Optional mode/shape/padding flags and fill values (e.g. 'same', 'full', 'valid', 'replicate', 'symmetric', 'circular', 'fill', padval, 'conv', 'corr').",
    },
];

const IMFILTER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = imfilter(A, H)",
        inputs: &IMFILTER_INPUTS_CORE,
        outputs: &IMFILTER_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = imfilter(A, H, options...)",
        inputs: &IMFILTER_INPUTS_WITH_OPTIONS,
        outputs: &IMFILTER_OUTPUT,
    },
];

const IMFILTER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INVALID_INPUT",
    identifier: Some("RunMat:imfilter:InvalidInput"),
    when: "Input/kernel tensors are invalid or cannot be converted.",
    message: "imfilter: invalid input",
};

const IMFILTER_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INVALID_OPTION",
    identifier: Some("RunMat:imfilter:InvalidOption"),
    when: "One or more option flags or option values are invalid.",
    message: "imfilter: invalid option",
};

const IMFILTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMFILTER.INTERNAL",
    identifier: Some("RunMat:imfilter:Internal"),
    when: "Internal filtering operation fails.",
    message: "imfilter: internal operation failed",
};

const IMFILTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMFILTER_ERROR_INVALID_INPUT,
    IMFILTER_ERROR_INVALID_OPTION,
    IMFILTER_ERROR_INTERNAL,
];

pub const IMFILTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMFILTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMFILTER_ERRORS,
};

const IMFILTER_VALID_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imfilter-valid-output-shape",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "imfilter(..., 'valid') is a RunMat-only output-shape extension",
    error_identifier: Some("RunMat:compatibility:ImfilterValidExtension"),
};

const IMFILTER_FILL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "imfilter-fill-keyword",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "imfilter(..., 'fill', padval) is a RunMat-only spelling; MATLAB accepts padval directly",
    error_identifier: Some("RunMat:compatibility:ImfilterFillExtension"),
};

pub const IMFILTER_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [IMFILTER_VALID_EXTENSION, IMFILTER_FILL_EXTENSION];

const IMFILTER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented int8, int16, int32, uint8, uint16, and uint32 images retain their authoritative class. Host and automatic-fallback calls accumulate in binary64; explicit gpuArray calls accumulate in binary64 for uint32 and binary32 for the other documented integer classes.",
    }];

pub const IMFILTER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = imfilter(integer_A, double_h, options...)" ,
        inputs: &IMFILTER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host and automatic-placement fallback sums use binary64 so planner placement is transparent. Explicit gpuArray uint32 sums use binary64, while the other documented explicit resident integer classes use binary32 multiplication and accumulation. Fractional results round to nearest and out-of-range results saturate before authoritative same-class storage is constructed; owner-aware fallback preserves explicit residency.",
    }];

fn filter_error(builtin: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

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
        imfilter_error_with_detail(fallback, err.message())
    }
}

#[runtime_builtin(
    name = "imfilter",
    category = "image/filters",
    summary = "Apply linear image filters.",
    keywords = "imfilter,image,filter,convolution,correlation,padding",
    accel = "custom-imfilter",
    type_resolver(imfilter_type),
    descriptor(crate::builtins::image::filters::imfilter::IMFILTER_DESCRIPTOR),
    extensions(crate::builtins::image::filters::imfilter::IMFILTER_EXTENSIONS),
    integer_capabilities(crate::builtins::image::filters::imfilter::IMFILTER_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::filters::imfilter"
)]
async fn imfilter_builtin(
    image: Value,
    kernel: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let options = parse_imfilter_options(&rest)?;
    match (image, kernel) {
        (Value::GpuTensor(image_handle), Value::GpuTensor(filter_handle)) => {
            imfilter_gpu(image_handle, Value::GpuTensor(filter_handle), options).await
        }
        (Value::GpuTensor(image_handle), filter_value) => {
            imfilter_gpu(image_handle, filter_value, options).await
        }
        (image_value, Value::GpuTensor(filter_handle)) => {
            let filter_tensor = download_kernel_tensor(&filter_handle).await?;
            imfilter_host_value(image_value, filter_tensor, options)
        }
        (image_value, filter_value) => {
            let filter_tensor = host_kernel_tensor(filter_value)?;
            imfilter_host_value(image_value, filter_tensor, options)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageOutputClass {
    Double,
    Single,
    Integer(NumericDType),
    Logical,
}

#[derive(Clone)]
struct ProtectedGpuMetadata {
    numeric: crate::builtins::math::fft::common::GpuMetadataSnapshot,
    class_name: Option<String>,
    transpose: Option<runmat_accelerate_api::TransposeInfo>,
}

fn protected_gpu_metadata(handle: &GpuTensorHandle) -> ProtectedGpuMetadata {
    ProtectedGpuMetadata {
        numeric: gpu_metadata_snapshot(handle),
        class_name: runmat_accelerate_api::handle_class_name(handle),
        transpose: runmat_accelerate_api::handle_transpose_info(handle),
    }
}

fn restore_protected_gpu_metadata(handle: &GpuTensorHandle, metadata: ProtectedGpuMetadata) {
    restore_gpu_metadata(handle, metadata.numeric);
    match metadata.class_name {
        Some(class_name) => runmat_accelerate_api::set_handle_class_name(handle, class_name),
        None => runmat_accelerate_api::clear_handle_class_name(handle),
    }
    match metadata.transpose {
        Some(transpose) => runmat_accelerate_api::record_handle_transpose(
            handle,
            transpose.base_rows,
            transpose.base_cols,
        ),
        None => runmat_accelerate_api::clear_handle_transpose(handle),
    }
    runmat_accelerate_api::mark_residency(handle);
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
    if tensor.is_empty() || tensor.shape.contains(&0) {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be non-empty",
        ));
    }
    if tensor.shape.len() > 2 {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H must be a double vector or matrix",
        ));
    }
    Ok(())
}

async fn download_kernel_tensor(handle: &GpuTensorHandle) -> BuiltinResult<Tensor> {
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
    let provider = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "no acceleration provider owns H",
        )
    })?;
    if provider.precision() != ProviderPrecision::F64 {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "H is labelled double but its owner cannot physically represent double storage",
        ));
    }
    let metadata = gpu_helpers::snapshot_handle_metadata(handle);
    let result = gpu_helpers::download_value_preserving_residency_async(provider, handle).await;
    gpu_helpers::restore_handle_metadata(handle, &metadata);
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

async fn imfilter_gpu(
    image_handle: GpuTensorHandle,
    kernel_value: Value,
    options: ImfilterOptions,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        let kernel_is_wgpu = matches!(kernel_value, Value::GpuTensor(ref h) if h.device_id != 0);
        if kernel_is_wgpu || image_handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let output_class = ImageOutputClass::from_gpu_handle(&image_handle)?;
    let provider = gpu_helpers::exact_provider_for_handle(&image_handle).ok_or_else(|| {
        imfilter_error_with_detail(&IMFILTER_ERROR_INTERNAL, "no acceleration provider owns A")
    })?;
    if output_class
        .provider_precision()
        .is_some_and(|precision| precision != provider.precision())
    {
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INVALID_INPUT,
            "A GPU precision metadata contradicts its owning provider",
        ));
    }

    let (kernel_tensor, resident_kernel) = match kernel_value {
        Value::GpuTensor(handle) => (download_kernel_tensor(&handle).await?, Some(handle)),
        other => (host_kernel_tensor(other)?, None),
    };
    let expected_shape = build_imfilter_plan(
        &image_handle.shape,
        &kernel_tensor,
        &options,
        IMFILTER_BUILTIN,
    )?
    .final_shape;

    // Native providers currently expose floating imfilter kernels. Integer and
    // logical images take an exact owner-aware host path, then explicit inputs
    // are restored to their owner with their original class metadata.
    let Some(expected_precision) = output_class.provider_precision() else {
        return imfilter_gpu_host_fallback(provider, &image_handle, &kernel_tensor, &options).await;
    };
    let mut uploaded_kernel = None;
    let kernel_handle = if let Some(handle) = resident_kernel {
        let same_owner = gpu_helpers::exact_provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider));
        if same_owner {
            handle
        } else {
            let uploaded =
                match upload_kernel(provider, &kernel_tensor, &image_handle, Some(&handle)) {
                    Ok(uploaded) => uploaded,
                    Err(_) => {
                        return imfilter_gpu_host_fallback(
                            provider,
                            &image_handle,
                            &kernel_tensor,
                            &options,
                        )
                        .await
                    }
                };
            uploaded_kernel = Some(uploaded.clone());
            uploaded
        }
    } else {
        let uploaded = match upload_kernel(provider, &kernel_tensor, &image_handle, None) {
            Ok(uploaded) => uploaded,
            Err(_) => {
                return imfilter_gpu_host_fallback(
                    provider,
                    &image_handle,
                    &kernel_tensor,
                    &options,
                )
                .await
            }
        };
        uploaded_kernel = Some(uploaded.clone());
        uploaded
    };

    let image_metadata = protected_gpu_metadata(&image_handle);
    let kernel_metadata = protected_gpu_metadata(&kernel_handle);
    let provider_result = provider
        .imfilter(&image_handle, &kernel_handle, &options)
        .await;
    restore_protected_gpu_metadata(&image_handle, image_metadata);
    restore_protected_gpu_metadata(&kernel_handle, kernel_metadata);
    if let Some(uploaded) = uploaded_kernel.as_ref() {
        free_temporary_handle(provider, uploaded);
    }

    match provider_result {
        Ok(mut output) => {
            if !valid_provider_output(
                &output,
                &image_handle,
                &kernel_handle,
                &expected_shape,
                expected_precision,
                provider,
            ) {
                free_rejected_provider_output(provider, &output, &image_handle, &kernel_handle);
                return Err(imfilter_error_with_detail(
                    &IMFILTER_ERROR_INTERNAL,
                    "provider returned an invalid imfilter result",
                ));
            }
            runmat_accelerate_api::set_handle_provenance(
                &mut output,
                runmat_accelerate_api::handle_provenance(&image_handle)
                    .unwrap_or(GpuHandleProvenance::Automatic),
            );
            runmat_accelerate_api::mark_residency(&output);
            Ok(Value::GpuTensor(output))
        }
        Err(error) => {
            log::trace!("imfilter: provider path unavailable, using host fallback: {error}");
            imfilter_gpu_host_fallback(provider, &image_handle, &kernel_tensor, &options).await
        }
    }
}

fn upload_kernel(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
    image: &GpuTensorHandle,
    resident_kernel: Option<&GpuTensorHandle>,
) -> BuiltinResult<GpuTensorHandle> {
    let values = tensor.as_f64_slice().ok_or_else(|| {
        imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, "H must be double")
    })?;
    let image_metadata = protected_gpu_metadata(image);
    let resident_kernel_metadata = resident_kernel.map(protected_gpu_metadata);
    let upload_result = provider.upload(&HostTensorView {
        data: values,
        shape: &tensor.shape,
    });
    restore_protected_gpu_metadata(image, image_metadata);
    if let (Some(kernel), Some(metadata)) = (resident_kernel, resident_kernel_metadata) {
        restore_protected_gpu_metadata(kernel, metadata);
    }
    let output = upload_result.map_err(|error| {
        imfilter_error_with_detail(
            &IMFILTER_ERROR_INTERNAL,
            format!("failed to upload H to A's provider: {error}"),
        )
    })?;
    let aliases_image = same_handle(&output, image);
    let aliases_resident_kernel =
        resident_kernel.is_some_and(|kernel| same_handle(&output, kernel));
    let valid = output.shape == tensor.shape
        && output.device_id == provider.device_id()
        && !aliases_image
        && !aliases_resident_kernel
        && runmat_accelerate_api::provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(provider.precision())
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output)
        && gpu_helpers::gpu_class_metadata_matches(
            &output,
            Some(provider.precision()),
            None,
            false,
        );
    if !valid {
        if !aliases_image && !aliases_resident_kernel {
            free_actual_owner(&output, "invalid uploaded kernel");
        }
        return Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INTERNAL,
            "provider returned an invalid uploaded H handle",
        ));
    }
    Ok(output)
}

fn same_handle(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

async fn imfilter_gpu_host_fallback(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    image_handle: &GpuTensorHandle,
    kernel: &Tensor,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::snapshot_handle_metadata(image_handle);
    let download =
        gpu_helpers::download_value_preserving_residency_async(provider, image_handle).await;
    gpu_helpers::restore_handle_metadata(image_handle, &metadata);
    let image = download.map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    let output_class = ImageOutputClass::from_gpu_handle(image_handle)?;
    let image_tensor = tensor::value_into_tensor_for(IMFILTER_BUILTIN, image)
        .map_err(|err| imfilter_error_with_detail(&IMFILTER_ERROR_INVALID_INPUT, err))?;
    let accumulation = fallback_accumulation(
        output_class,
        runmat_accelerate_api::handle_is_explicit(image_handle),
    );
    let result = apply_imfilter_host(
        &image_tensor,
        kernel,
        options,
        output_class,
        accumulation,
        IMFILTER_BUILTIN,
    )
    .map_err(|err| imfilter_map_error(err, &IMFILTER_ERROR_INVALID_INPUT))?;
    if !runmat_accelerate_api::handle_is_explicit(image_handle) {
        return Ok(result);
    }
    let restored =
        gpu_helpers::restore_class_preserving_value(image_handle, result, IMFILTER_BUILTIN)?;
    if matches!(restored, Value::GpuTensor(_)) {
        Ok(restored)
    } else {
        Err(imfilter_error_with_detail(
            &IMFILTER_ERROR_INTERNAL,
            "explicit gpuArray result could not be restored to its owning provider",
        ))
    }
}

fn fallback_accumulation(
    output_class: ImageOutputClass,
    explicit_gpu_array: bool,
) -> FilterAccumulation {
    if !explicit_gpu_array {
        FilterAccumulation::HostDouble
    } else {
        match output_class {
            ImageOutputClass::Integer(NumericDType::U32) | ImageOutputClass::Double => {
                FilterAccumulation::HostDouble
            }
            ImageOutputClass::Integer(_) | ImageOutputClass::Logical | ImageOutputClass::Single => {
                FilterAccumulation::GpuSingle
            }
        }
    }
}

fn valid_provider_output(
    output: &GpuTensorHandle,
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
    expected_shape: &[usize],
    expected_precision: ProviderPrecision,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    output.shape == expected_shape
        && output.device_id == image.device_id
        && !(output.device_id == image.device_id && output.buffer_id == image.buffer_id)
        && !(output.device_id == kernel.device_id && output.buffer_id == kernel.buffer_id)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, Some(expected_precision), None, false)
}

fn free_temporary_handle(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) {
    let Some(owner) = gpu_helpers::exact_provider_for_handle(handle) else {
        log::trace!("imfilter: temporary kernel has no registered owner");
        return;
    };
    if !std::ptr::eq(owner, provider) {
        log::trace!("imfilter: temporary kernel owner changed before cleanup");
        return;
    }
    if let Err(error) = owner.free(handle) {
        log::trace!("imfilter: failed to free temporary kernel: {error}");
    } else {
        runmat_accelerate_api::clear_handle_metadata(handle);
    }
}

fn free_actual_owner(handle: &GpuTensorHandle, context: &str) {
    let Some(owner) = gpu_helpers::exact_provider_for_handle(handle) else {
        log::trace!("imfilter: {context} has no registered owner");
        return;
    };
    if let Err(error) = owner.free(handle) {
        log::trace!("imfilter: failed to free {context}: {error}");
    } else {
        runmat_accelerate_api::clear_handle_metadata(handle);
    }
}

fn free_rejected_provider_output(
    _invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: &GpuTensorHandle,
    image: &GpuTensorHandle,
    kernel: &GpuTensorHandle,
) {
    let aliases_protected = (output.device_id == image.device_id
        && output.buffer_id == image.buffer_id)
        || (output.device_id == kernel.device_id && output.buffer_id == kernel.buffer_id);
    if aliases_protected {
        return;
    }
    let Some(owner) = gpu_helpers::exact_provider_for_handle(output) else {
        log::trace!("imfilter: rejected provider result has no registered owner");
        return;
    };
    if let Err(error) = owner.free(output) {
        log::trace!("imfilter: failed to free rejected provider result: {error}");
    } else {
        runmat_accelerate_api::clear_handle_metadata(output);
    }
}

fn parse_imfilter_options(args: &[Value]) -> BuiltinResult<ImfilterOptions> {
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
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "replicate" => options.padding = ImfilterPadding::Replicate,
                "symmetric" => options.padding = ImfilterPadding::Symmetric,
                "circular" => options.padding = ImfilterPadding::Circular,
                "fill" => {
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
                "same" => options.shape = ImfilterShape::Same,
                "full" => options.shape = ImfilterShape::Full,
                "valid" => {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &IMFILTER_VALID_EXTENSION,
                        IMFILTER_BUILTIN,
                    )?;
                    options.shape = ImfilterShape::Valid;
                }
                "conv" => options.mode = ImfilterMode::Convolution,
                "corr" => options.mode = ImfilterMode::Correlation,
                other => {
                    return Err(imfilter_error_with_detail(
                        &IMFILTER_ERROR_INVALID_OPTION,
                        format!(
                            "unknown option '{}' (supported: 'same', 'full', 'valid', 'replicate', 'symmetric', 'circular', 'fill', 'conv', 'corr')",
                            other
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

/// Core host implementation of `imfilter`, shared with the in-process acceleration provider.
#[derive(Clone, Debug)]
pub struct ImfilterKernelPoint {
    pub offsets: Vec<isize>,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct ImfilterPlan {
    pub rank: usize,
    pub output_shape_ext: Vec<usize>,
    pub final_shape: Vec<usize>,
    pub image_shape_ext: Vec<usize>,
    pub image_strides: Vec<usize>,
    pub base_offset: Vec<isize>,
    pub kernel_points: Vec<ImfilterKernelPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterAccumulation {
    HostDouble,
    GpuSingle,
}

impl ImfilterPlan {
    #[inline]
    pub fn evaluate(&self, image_data: &[f64], options: &ImfilterOptions) -> Vec<f64> {
        self.evaluate_with_accumulation(image_data, options, FilterAccumulation::HostDouble)
    }

    fn evaluate_with_accumulation(
        &self,
        image_data: &[f64],
        options: &ImfilterOptions,
        accumulation: FilterAccumulation,
    ) -> Vec<f64> {
        evaluate_filter(
            &self.output_shape_ext,
            &self.base_offset,
            &self.image_shape_ext,
            &self.image_strides,
            &self.kernel_points,
            options,
            image_data,
            accumulation,
        )
    }
}

pub fn build_imfilter_plan(
    image_shape: &[usize],
    kernel: &Tensor,
    options: &ImfilterOptions,
    builtin: &str,
) -> BuiltinResult<ImfilterPlan> {
    if kernel.is_empty() || kernel.shape.contains(&0) {
        return Err(filter_error(
            builtin,
            format!("{builtin}: filter must be non-empty along every dimension"),
        ));
    }

    let image_shape_norm = normalize_shape(image_shape);
    let kernel_shape_norm = normalize_shape(&kernel.shape);
    let rank = image_shape_norm.len().max(kernel_shape_norm.len());
    let image_ext = extend_shape(&image_shape_norm, rank);
    let kernel_ext = extend_shape(&kernel_shape_norm, rank);

    validate_kernel_shape(&image_shape_norm, &kernel_ext, builtin)?;

    // MATLAB defines the one-based center as floor((size(h) + 1) / 2).
    // In zero-based coordinates that is (size(h) - 1) / 2, which matters for
    // every even-sized kernel.
    let origin: Vec<usize> = kernel_ext
        .iter()
        .map(|&dim| dim.saturating_sub(1) / 2)
        .collect();
    let full_shape: Vec<usize> = image_ext
        .iter()
        .zip(kernel_ext.iter())
        .map(|(&img, &ker)| img + ker - 1)
        .collect();

    let kernel_points = build_kernel_points(kernel, &kernel_ext, &origin, options.mode);
    let zero_offset = vec![0isize; rank];
    let origin_signed: Vec<isize> = origin.iter().map(|&o| o as isize).collect();
    let full_offset: Vec<isize> = origin
        .iter()
        .zip(kernel_ext.iter())
        .map(|(&orig, &dim)| orig as isize - (dim as isize - 1))
        .collect();

    let (target_shape_ext, base_offset) = match options.shape {
        ImfilterShape::Full => (full_shape.clone(), full_offset),
        ImfilterShape::Same => (image_ext.clone(), zero_offset),
        ImfilterShape::Valid => {
            let target: Vec<usize> = image_ext
                .iter()
                .zip(kernel_ext.iter())
                .map(|(&img, &ker)| if img >= ker { img - ker + 1 } else { 0 })
                .collect();
            (target, origin_signed)
        }
    };

    let mut final_shape = target_shape_ext.clone();
    while final_shape.len() > image_shape_norm.len() && final_shape.last() == Some(&1) {
        final_shape.pop();
    }
    if final_shape.is_empty() {
        final_shape.push(1);
    }

    let image_strides = compute_strides(&image_ext);

    Ok(ImfilterPlan {
        rank,
        output_shape_ext: target_shape_ext,
        final_shape,
        image_shape_ext: image_ext,
        image_strides,
        base_offset,
        kernel_points,
    })
}

pub fn apply_imfilter_tensor(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    builtin: &str,
) -> BuiltinResult<Tensor> {
    let plan = build_imfilter_plan(&image.shape, kernel, options, builtin)?;
    let image_values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate(&image_values, options);
    Tensor::new(data, plan.final_shape.clone())
        .map_err(|e| filter_error(builtin, format!("{builtin}: {e}")))
}

fn apply_imfilter_host(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    output_class: ImageOutputClass,
    accumulation: FilterAccumulation,
    builtin: &str,
) -> BuiltinResult<Value> {
    let plan = build_imfilter_plan(&image.shape, kernel, options, builtin)?;
    let image_values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate_with_accumulation(&image_values, options, accumulation);
    let shape = plan.final_shape;

    let storage = match output_class {
        ImageOutputClass::Double => NumericStorage::F64(data),
        ImageOutputClass::Single => {
            NumericStorage::F32(data.into_iter().map(|value| value as f32).collect())
        }
        ImageOutputClass::Integer(dtype) => integer_filter_storage(data, dtype),
        ImageOutputClass::Logical => {
            let bits = data
                .into_iter()
                .map(|value| u8::from(round_and_clamp(value, 0.0, 1.0) != 0.0))
                .collect();
            let logical = LogicalArray::new(bits, shape)
                .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
            return Ok(if logical.data.len() == 1 {
                Value::Bool(logical.data[0] != 0)
            } else {
                Value::LogicalArray(logical)
            });
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn integer_filter_storage(data: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    macro_rules! signed {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| {
                        round_and_clamp(value, <$ty>::MIN as f64, <$ty>::MAX as f64) as $ty
                    })
                    .collect(),
            )
        };
    }
    macro_rules! unsigned {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| round_and_clamp(value, 0.0, <$ty>::MAX as f64) as $ty)
                    .collect(),
            )
        };
    }
    match dtype {
        NumericDType::I8 => signed!(I8, i8),
        NumericDType::I16 => signed!(I16, i16),
        NumericDType::I32 => signed!(I32, i32),
        NumericDType::U8 => unsigned!(U8, u8),
        NumericDType::U16 => unsigned!(U16, u16),
        NumericDType::U32 => unsigned!(U32, u32),
        NumericDType::I64 | NumericDType::U64 | NumericDType::F32 | NumericDType::F64 => {
            unreachable!("unsupported imfilter output class rejected before finalization")
        }
    }
}

fn round_and_clamp(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0_f64.clamp(minimum, maximum)
    } else {
        value.round().clamp(minimum, maximum)
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn validate_kernel_shape(
    image_shape: &[usize],
    kernel_ext: &[usize],
    builtin: &str,
) -> BuiltinResult<()> {
    for (dim_idx, &ker_dim) in kernel_ext.iter().enumerate() {
        let img_dim = image_shape.get(dim_idx).copied().unwrap_or(1);
        if dim_idx >= image_shape.len() && ker_dim > 1 {
            return Err(filter_error(
                builtin,
                format!(
                    "{builtin}: filter dimension {} is {}, but the image has no corresponding axis",
                    dim_idx + 1,
                    ker_dim
                ),
            ));
        }
        if img_dim == 0 {
            return Err(filter_error(
                builtin,
                format!("{builtin}: image must not have zero-length dimensions"),
            ));
        }
    }
    Ok(())
}

fn extend_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    while out.len() < rank {
        out.push(1);
    }
    out
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn advance_index(index: &mut [usize], shape: &[usize]) {
    for (idx, &dimension) in index.iter_mut().zip(shape.iter()) {
        *idx += 1;
        if *idx < dimension {
            return;
        }
        *idx = 0;
    }
}

fn build_kernel_points(
    kernel: &Tensor,
    kernel_shape: &[usize],
    origin: &[usize],
    mode: ImfilterMode,
) -> Vec<ImfilterKernelPoint> {
    let rank = kernel_shape.len();
    let strides = compute_strides(kernel_shape);
    let values = tensor::tensor_values_f64_cow(kernel);
    let total = values.len();
    let mut points = Vec::with_capacity(total);
    if total == 0 {
        return points;
    }

    let mut index = vec![0usize; rank];
    for _ in 0..total {
        let linear = index_to_linear(&index, &strides);
        let value = match mode {
            ImfilterMode::Correlation => values[linear],
            ImfilterMode::Convolution => {
                values[flipped_linear_index(&index, kernel_shape, &strides)]
            }
        };
        let offsets = index
            .iter()
            .zip(origin.iter())
            .map(|(&idx, &orig)| idx as isize - orig as isize)
            .collect();
        points.push(ImfilterKernelPoint { offsets, value });
        advance_index(&mut index, kernel_shape);
    }

    points
}

fn flipped_linear_index(index: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .enumerate()
        .map(|(dim, &coord)| (shape[dim] - 1 - coord) * strides[dim])
        .sum()
}

fn index_to_linear(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

fn sample_with_padding(
    image: &[f64],
    image_shape: &[usize],
    image_strides: &[usize],
    base_index: &[usize],
    base_offset: &[isize],
    offsets: &[isize],
    options: &ImfilterOptions,
) -> f64 {
    if image.is_empty() {
        return options.constant_value;
    }

    let mut final_indices = Vec::with_capacity(image_shape.len());
    for (dim, (&base, &offset)) in base_index.iter().zip(offsets.iter()).enumerate() {
        let coord = base as isize + base_offset[dim] + offset;
        let len = image_shape[dim] as isize;
        if coord >= 0 && coord < len {
            final_indices.push(coord as usize);
            continue;
        }
        match options.padding {
            ImfilterPadding::Constant => return options.constant_value,
            ImfilterPadding::Replicate => final_indices.push(clamp_index(coord, len)),
            ImfilterPadding::Circular => final_indices.push(wrap_index(coord, len)),
            ImfilterPadding::Symmetric => final_indices.push(reflect_index(coord, len)),
        }
    }

    let linear: usize = final_indices
        .iter()
        .zip(image_strides.iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum();
    image.get(linear).copied().unwrap_or(options.constant_value)
}

fn evaluate_filter(
    output_shape: &[usize],
    base_offset: &[isize],
    image_shape: &[usize],
    image_strides: &[usize],
    kernel_points: &[ImfilterKernelPoint],
    options: &ImfilterOptions,
    image_data: &[f64],
    accumulation: FilterAccumulation,
) -> Vec<f64> {
    let total = tensor::element_count(output_shape);
    let mut out = vec![0.0; total];
    if total == 0 || kernel_points.is_empty() {
        return out;
    }

    let mut out_index = vec![0usize; output_shape.len()];
    for out_value in out.iter_mut() {
        *out_value = match accumulation {
            FilterAccumulation::HostDouble => {
                let mut sum = 0.0;
                for point in kernel_points {
                    let value = sample_with_padding(
                        image_data,
                        image_shape,
                        image_strides,
                        &out_index,
                        base_offset,
                        &point.offsets,
                        options,
                    );
                    sum += point.value * value;
                }
                sum
            }
            FilterAccumulation::GpuSingle => {
                let mut sum = 0.0_f32;
                for point in kernel_points {
                    let value = sample_with_padding(
                        image_data,
                        image_shape,
                        image_strides,
                        &out_index,
                        base_offset,
                        &point.offsets,
                        options,
                    ) as f32;
                    sum += (point.value as f32) * value;
                }
                f64::from(sum)
            }
        };
        advance_index(&mut out_index, output_shape);
    }

    out
}

fn clamp_index(coord: isize, len: isize) -> usize {
    if len <= 0 || coord <= 0 {
        0
    } else if coord >= len {
        (len - 1) as usize
    } else {
        coord as usize
    }
}

fn wrap_index(mut coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    coord %= len;
    if coord < 0 {
        coord += len;
    }
    coord as usize
}

fn reflect_index(coord: isize, len: isize) -> usize {
    if len <= 0 {
        return 0;
    }
    if len == 1 {
        return 0;
    }
    // MATLAB's symmetric boundary repeats the border sample. For [a b c],
    // the conceptual extension is ... c b a a b c c b a ... .
    let period = 2 * len;
    let mut value = coord % period;
    if value < 0 {
        value += period;
    }
    if value >= len {
        value = period - 1 - value;
    }
    value as usize
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{AccelDownloadFuture, AccelProvider, AccelProviderFuture};
    use runmat_builtins::{IntegerStorage, Tensor, Value};

    struct MutatingAliasingImfilterProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
    }

    impl MutatingAliasingImfilterProvider {
        fn new() -> Self {
            Self {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            }
        }
    }

    impl AccelProvider for MutatingAliasingImfilterProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.inner.download(handle)
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            self.inner.free(handle)
        }

        fn device_info(&self) -> String {
            "mutating-aliasing-imfilter-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }

        fn imfilter<'a>(
            &'a self,
            image: &'a GpuTensorHandle,
            kernel: &'a GpuTensorHandle,
            _options: &'a ImfilterOptions,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            runmat_accelerate_api::set_handle_precision(image, ProviderPrecision::F32);
            runmat_accelerate_api::set_handle_storage(image, GpuTensorStorage::ComplexInterleaved);
            runmat_accelerate_api::set_handle_integer_type(image, IntegerElementType::U8);
            runmat_accelerate_api::set_handle_logical(image, true);
            runmat_accelerate_api::set_handle_class_name(image, "single");
            runmat_accelerate_api::clear_handle_transpose(image);
            runmat_accelerate_api::set_handle_precision(kernel, ProviderPrecision::F32);
            Box::pin(async move { Ok(image.clone()) })
        }
    }

    fn simple_tensor(data: &[f64], rows: usize, cols: usize) -> Tensor {
        Tensor::new(data.to_vec(), vec![rows, cols]).unwrap()
    }

    fn typed_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor
    }

    fn error_message(err: &crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn same_padding_default_zero() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let options = ImfilterOptions::default();
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        assert!(result
            .materialize_f64()
            .iter()
            .all(|&v| (v - 10.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replicate_padding() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let options = ImfilterOptions {
            padding: ImfilterPadding::Replicate,
            ..Default::default()
        };
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        let expected = [18.0, 24.0, 21.0, 27.0];
        for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn full_output_matches_expected_size() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let options = ImfilterOptions {
            shape: ImfilterShape::Full,
            ..Default::default()
        };
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        assert_eq!(result.shape, vec![3, 3]);
        let expected = [4.0, 14.0, 6.0, 11.0, 30.0, 11.0, 6.0, 14.0, 4.0];
        for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn valid_output_respects_kernel_size() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 4], 2, 2);
        let options = ImfilterOptions {
            shape: ImfilterShape::Valid,
            ..Default::default()
        };
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        assert_eq!(result.shape, vec![1, 1]);
        assert!((result.materialize_f64()[0] - 10.0).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn convolution_matches_correlation_with_flipped_kernel() {
        let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 3, 2);
        let kernel = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let corr_opts = ImfilterOptions::default();
        let corr =
            apply_imfilter_tensor(&image, &kernel, &corr_opts, IMFILTER_BUILTIN).expect("corr");
        let flipped_kernel = simple_tensor(&[4.0, 3.0, 2.0, 1.0], 2, 2);
        let corr_flipped =
            apply_imfilter_tensor(&image, &flipped_kernel, &corr_opts, IMFILTER_BUILTIN)
                .expect("corr flip");
        let conv_opts = ImfilterOptions {
            mode: ImfilterMode::Convolution,
            ..Default::default()
        };
        let conv =
            apply_imfilter_tensor(&image, &kernel, &conv_opts, IMFILTER_BUILTIN).expect("conv");
        assert_eq!(conv.shape, corr_flipped.shape);
        for ((a, b), c) in conv
            .materialize_f64()
            .iter()
            .zip(corr_flipped.materialize_f64().iter())
            .zip(corr.materialize_f64().iter())
        {
            assert!((a - b).abs() < 1e-12 || (a - c).abs() < 1e-8);
        }
    }

    #[test]
    fn raw_filter_plan_reads_typed_integer_image_storage_exactly() {
        let image = typed_tensor(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]);
        let kernel = simple_tensor(&[2.0], 1, 1);

        let result = apply_imfilter_tensor(
            &image,
            &kernel,
            &ImfilterOptions::default(),
            IMFILTER_BUILTIN,
        )
        .expect("imfilter");

        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.materialize_f64(), vec![2.0, 6.0, 4.0, 8.0]);
    }

    #[test]
    fn resident_integer_fallback_uses_documented_binary32_accumulation() {
        let image = simple_tensor(&[16_777_216.0, 1.0, -16_777_216.0], 3, 1);
        let kernel = simple_tensor(&[1.0, 1.0, 1.0], 3, 1);
        let options = ImfilterOptions {
            shape: ImfilterShape::Valid,
            ..Default::default()
        };
        let plan = build_imfilter_plan(&image.shape, &kernel, &options, IMFILTER_BUILTIN)
            .expect("filter plan");
        let values = image.materialize_f64();
        assert_eq!(
            plan.evaluate_with_accumulation(&values, &options, FilterAccumulation::HostDouble),
            vec![1.0]
        );
        assert_eq!(
            plan.evaluate_with_accumulation(&values, &options, FilterAccumulation::GpuSingle),
            vec![0.0]
        );
        assert_eq!(
            fallback_accumulation(ImageOutputClass::Integer(NumericDType::U8), false),
            FilterAccumulation::HostDouble,
            "automatic placement must remain numerically transparent"
        );
        assert_eq!(
            fallback_accumulation(ImageOutputClass::Integer(NumericDType::U8), true),
            FilterAccumulation::GpuSingle,
            "explicit gpuArray follows the documented GPU precision"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn circular_padding_wraps_indices() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[0.0, 1.0, 1.0, 0.0], 2, 2);
        let options = ImfilterOptions {
            padding: ImfilterPadding::Circular,
            ..Default::default()
        };
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        let expected = [5.0, 5.0, 5.0, 5.0];
        for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_fallback_uses_provider_upload() {
        test_support::with_test_provider(|provider| {
            let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0], 2, 2);
            let kernel = simple_tensor(&[1.0, 1.0, 1.0, 1.0], 2, 2);
            let image_view = HostTensorView {
                data: &image.materialize_f64(),
                shape: &image.shape,
            };
            let kernel_view = HostTensorView {
                data: &kernel.materialize_f64(),
                shape: &kernel.shape,
            };
            let image_handle = provider.upload(&image_view).expect("upload image");
            let kernel_handle = provider.upload(&kernel_view).expect("upload kernel");
            let value = block_on(imfilter_builtin(
                Value::GpuTensor(image_handle),
                Value::GpuTensor(kernel_handle),
                Vec::new(),
            ))
            .expect("imfilter");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            // A 2-by-2 kernel has its documented origin at the first element,
            // so same-size correlation samples forward from each output site.
            let expected = [12.0, 9.0, 7.0, 5.0];
            for (got, exp) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((got - exp).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn public_gpu_form_rejects_non_double_kernel_before_upload() {
        test_support::with_test_provider(|provider| {
            let image = simple_tensor(&[1.0, 4.0, 2.0, 5.0], 2, 2);
            let image_view = HostTensorView {
                data: &image.materialize_f64(),
                shape: &image.shape,
            };
            let image_handle = provider.upload(&image_view).expect("upload image");
            let kernel = typed_tensor(IntegerStorage::I16(vec![2]), vec![1, 1]);

            let error = block_on(imfilter_builtin(
                Value::GpuTensor(image_handle),
                Value::Tensor(kernel),
                Vec::new(),
            ))
            .expect_err("typed H must reject");
            assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);
            assert!(error.message().contains("H must be double"));
        });
    }

    #[test]
    fn adversarial_provider_cannot_mutate_or_return_the_image_handle() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(MutatingAliasingImfilterProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }
        let image = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .expect("image upload");
        runmat_accelerate_api::set_handle_precision(&image, ProviderPrecision::F64);
        runmat_accelerate_api::set_handle_storage(&image, GpuTensorStorage::Real);
        runmat_accelerate_api::set_handle_class_name(&image, "double");
        runmat_accelerate_api::record_handle_transpose(&image, 2, 2);
        let image = image.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);

        let error = block_on(imfilter_builtin(
            Value::GpuTensor(image.clone()),
            Value::Tensor(simple_tensor(&[1.0], 1, 1)),
            Vec::new(),
        ))
        .expect_err("aliased provider output must reject");

        assert_eq!(error.identifier(), IMFILTER_ERROR_INTERNAL.identifier);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&image),
            Some(ProviderPrecision::F64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_storage(&image),
            GpuTensorStorage::Real
        );
        assert_eq!(runmat_accelerate_api::handle_integer_type(&image), None);
        assert!(!runmat_accelerate_api::handle_is_logical(&image));
        assert_eq!(
            runmat_accelerate_api::handle_class_name(&image).as_deref(),
            Some("double")
        );
        assert_eq!(
            runmat_accelerate_api::handle_transpose_info(&image)
                .map(|info| { (info.base_rows, info.base_cols) }),
            Some((2, 2))
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&image));
        assert_eq!(
            block_on(provider.download(&image))
                .expect("protected input remains allocated")
                .data,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        provider.free(&image).expect("free image");
    }

    #[test]
    fn public_integer_images_preserve_every_documented_class() {
        let cases = [
            (
                IntegerStorage::I8(vec![-2, 63]),
                IntegerStorage::I8(vec![-4, 126]),
            ),
            (
                IntegerStorage::I16(vec![-2, 16_000]),
                IntegerStorage::I16(vec![-4, 32_000]),
            ),
            (
                IntegerStorage::I32(vec![-2, 1_000_000]),
                IntegerStorage::I32(vec![-4, 2_000_000]),
            ),
            (
                IntegerStorage::U8(vec![2, 127]),
                IntegerStorage::U8(vec![4, 254]),
            ),
            (
                IntegerStorage::U16(vec![2, 32_000]),
                IntegerStorage::U16(vec![4, 64_000]),
            ),
            (
                IntegerStorage::U32(vec![2, 1_000_000]),
                IntegerStorage::U32(vec![4, 2_000_000]),
            ),
        ];
        for (input, expected) in cases {
            let image = typed_tensor(input, vec![1, 2]);
            let result = block_on(imfilter_builtin(
                Value::Tensor(image),
                Value::Tensor(simple_tensor(&[2.0], 1, 1)),
                Vec::new(),
            ))
            .expect("integer imfilter");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor output");
            };
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn public_kernel_must_be_nonempty_double_vector_or_matrix() {
        let image = simple_tensor(&[1.0], 1, 1);
        let rank_three = Tensor::new(vec![1.0], vec![1, 1, 1]).unwrap();
        let error = block_on(imfilter_builtin(
            Value::Tensor(image.clone()),
            Value::Tensor(rank_three),
            Vec::new(),
        ))
        .expect_err("rank-three H must reject");
        assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);

        let empty = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let error = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(empty),
            Vec::new(),
        ))
        .expect_err("empty H must reject");
        assert_eq!(error.identifier(), IMFILTER_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn even_kernel_origin_and_symmetric_padding_match_public_rules() {
        let kernel = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let plan = build_imfilter_plan(
            &[1, 4],
            &kernel,
            &ImfilterOptions::default(),
            IMFILTER_BUILTIN,
        )
        .unwrap();
        let offsets: Vec<isize> = plan
            .kernel_points
            .iter()
            .map(|point| point.offsets[1])
            .collect();
        assert_eq!(offsets, vec![-1, 0, 1, 2]);
        assert_eq!(reflect_index(-1, 3), 0);
        assert_eq!(reflect_index(-2, 3), 1);
        assert_eq!(reflect_index(3, 3), 2);
        assert_eq!(reflect_index(4, 3), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_example_average_filter_matches_expected() {
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = Tensor::new(vec![1.0 / 9.0; 9], vec![3, 3]).unwrap();
        let result = apply_imfilter_tensor(
            &image,
            &kernel,
            &ImfilterOptions::default(),
            IMFILTER_BUILTIN,
        )
        .expect("imfilter");
        assert_eq!(result.shape, vec![2, 2]);
        for value in result.materialize_f64() {
            assert!((value - (10.0 / 9.0)).abs() < 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_example_convolution_same_matches_expected() {
        let image = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let kernel = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let options = ImfilterOptions {
            mode: ImfilterMode::Convolution,
            ..Default::default()
        };
        let result =
            apply_imfilter_tensor(&image, &kernel, &options, IMFILTER_BUILTIN).expect("imfilter");
        assert_eq!(result.shape, vec![3, 2]);
        // The documented even-kernel origin is the first element for a 2-by-2
        // kernel. Convolution rotates H while retaining that origin.
        let expected = [25.0, 35.0, 30.0, 26.0, 32.0, 24.0];
        for (got, exp) in result.materialize_f64().iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {got}, expected {exp}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn specifying_numeric_pad_value_matches_manual_options() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);

        let manual = ImfilterOptions {
            padding: ImfilterPadding::Constant,
            constant_value: 5.0,
            ..Default::default()
        };
        let manual_res =
            apply_imfilter_tensor(&image, &kernel, &manual, IMFILTER_BUILTIN).expect("imfilter");

        let via_builtin = block_on(imfilter_builtin(
            Value::Tensor(image.clone()),
            Value::Tensor(kernel.clone()),
            vec![Value::Num(5.0)],
        ))
        .expect("imfilter builtin");
        let via_tensor = tensor::value_into_tensor_for("imfilter", via_builtin).expect("tensor");
        assert_eq!(manual_res.shape, via_tensor.shape);
        for (a, b) in manual_res
            .materialize_f64()
            .iter()
            .zip(via_tensor.materialize_f64().iter())
        {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn parse_scalar_reads_typed_integer_tensor_exactly() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("scalar");
        assert_eq!(
            parse_scalar(IMFILTER_BUILTIN, &Value::Tensor(scalar)).unwrap(),
            u64::MAX as f64
        );

        let vector =
            Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).expect("vector");
        assert!(parse_scalar(IMFILTER_BUILTIN, &Value::Tensor(vector)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_option_string_raises_error() {
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let err = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("unsupported-mode")],
        ))
        .expect_err("imfilter should error");
        assert!(error_message(&err).contains("unknown option"));
        assert_eq!(err.identifier(), IMFILTER_ERROR_INVALID_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_requires_scalar_value() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 4], 2, 2);
        let pad = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("fill"), Value::Tensor(pad)],
        ))
        .expect_err("imfilter should error");
        assert!(error_message(&err).contains("scalar value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn valid_with_larger_kernel_returns_empty_tensor() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 25], 5, 5);
        let result = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("valid")],
        ))
        .expect("imfilter");
        match result {
            Value::Tensor(t) => {
                assert!(t.materialize_f64().is_empty());
                assert_eq!(t.shape, vec![0, 0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_without_value_defaults_to_zero_padding() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let image = simple_tensor(&[1.0, 3.0, 2.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0, 1.0, 1.0, 1.0], 2, 2);
        let default = block_on(imfilter_builtin(
            Value::Tensor(image.clone()),
            Value::Tensor(kernel.clone()),
            Vec::new(),
        ))
        .expect("imfilter default");
        let fill_only = block_on(imfilter_builtin(
            Value::Tensor(image),
            Value::Tensor(kernel),
            vec![Value::from("fill")],
        ))
        .expect("imfilter fill");
        assert_eq!(fill_only, default);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn imfilter_wgpu_matches_cpu_same_padding() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let image = simple_tensor(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let kernel = simple_tensor(&[1.0; 9], 3, 3);
        let cpu = apply_imfilter_tensor(
            &image,
            &kernel,
            &ImfilterOptions::default(),
            IMFILTER_BUILTIN,
        )
        .expect("cpu");

        let image_view = HostTensorView {
            data: &image.materialize_f64(),
            shape: &image.shape,
        };
        let image_handle = provider.upload(&image_view).expect("upload image");

        let gpu_value = block_on(imfilter_builtin(
            Value::GpuTensor(image_handle),
            Value::Tensor(kernel),
            Vec::new(),
        ))
        .expect("imfilter");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(cpu.shape, gathered.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in cpu
            .materialize_f64()
            .iter()
            .zip(gathered.materialize_f64().iter())
        {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }

    #[test]
    fn imfilter_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = IMFILTER_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"B = imfilter(A, H)"));
        assert!(labels.contains(&"B = imfilter(A, H, options...)"));
    }

    #[test]
    fn imfilter_extensions_are_gated_and_capability_covers_shape_changing_forms() {
        assert_eq!(IMFILTER_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            IMFILTER_INTEGER_CAPABILITIES[0].overload,
            BuiltinIntegerOverloadKind::Multiple
        );
        assert_eq!(
            IMFILTER_INTEGER_CAPABILITIES[0].overflow,
            BuiltinIntegerOverflowRule::Saturate
        );

        let image = Value::Tensor(simple_tensor(&[1.0], 1, 1));
        let kernel = Value::Tensor(simple_tensor(&[1.0], 1, 1));
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for option in ["valid", "fill"] {
            let error = block_on(imfilter_builtin(
                image.clone(),
                kernel.clone(),
                vec![Value::from(option)],
            ))
            .expect_err("extension must reject in MATLAB-compatible mode");
            assert!(error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")));
        }
    }

    #[test]
    fn imfilter_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = IMFILTER_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.IMFILTER.INVALID_INPUT"));
        assert!(codes.contains(&"RM.IMFILTER.INVALID_OPTION"));
        assert!(codes.contains(&"RM.IMFILTER.INTERNAL"));
    }
}
