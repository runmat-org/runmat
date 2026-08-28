//! MATLAB-compatible `ifftn` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, download_provider_complex_tensor, ensure_wide_integer_data_exact,
    free_rejected_provider_fft_output, gather_gpu_complex_tensor, gpu_metadata_snapshot,
    is_wide_integer_value, parse_nd_sizes_value, parse_symflag, provider_operation_unsupported,
    restore_complex_gpu_result, restore_gpu_metadata, restore_real_gpu_result, same_gpu_handle,
    transform_complex_tensor, transform_nd_complex_tensor, valid_provider_fft_output,
    value_to_complex_tensor, TransformDirection,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::fft::type_resolvers::ifftn_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_value::{ComplexTensor, Value};

use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifftn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifftn",
    op_kind: GpuOpKind::Custom("ifftn"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs sequential `ifft_dim` passes along each transformed axis; falls back to host execution when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifftn")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifftn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ifftn terminates fusion plans; fused kernels are not generated for N-D inverse FFTs.",
};

const BUILTIN_NAME: &str = "ifftn";

const IFFTN_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnWideIntegerDataExtension"),
};
const IFFTN_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn with int64 or uint64 SIZE controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnWideIntegerControlExtension"),
};
const IFFTN_SHORT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-short-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn SIZE vectors shorter than ndims(X) are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnShortSizeExtension"),
};
const IFFTN_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifftn-zero-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifftn SIZE vectors containing zero are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftnZeroSizeExtension"),
};
pub const IFFTN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    IFFTN_WIDE_DATA_EXTENSION,
    IFFTN_WIDE_CONTROL_EXTENSION,
    IFFTN_SHORT_SIZE_EXTENSION,
    IFFTN_ZERO_SIZE_EXTENSION,
];

const IFFTN_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented integer arrays enter the double N-dimensional inverse-FFT domain.",
}];
const IFFTN_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "SIZE is a positive integer vector with length at least ndims(X); logical vectors are documented.",
}];
const IFFTN_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Host wide data is gated and must be exactly representable as double.",
}];
const IFFTN_WIDE_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "Wide SIZE controls are independently gated and decoded exactly.",
}];
pub const IFFTN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftn(integer_X, ...)", inputs: &IFFTN_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer data produces double output; symmetric changes complexity only." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftn(X, integer_SIZE, ...)", inputs: &IFFTN_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Documented SIZE is decoded exactly and validated before execution." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftn(int64_or_uint64_X, ...)", inputs: &IFFTN_WIDE_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only wide host data may not silently round; resident wide data rejects before gather." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifftn(X, int64_or_uint64_SIZE, ...)", inputs: &IFFTN_WIDE_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only wide SIZE controls are independently gated; short and zero vectors have separate gates." },
];

const IFFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D inverse FFT output.",
}];

const IFFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

const IFFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform sizes per dimension.",
    },
];

const IFFTN_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFTN_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform sizes per dimension.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFTN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X)",
        inputs: &IFFTN_INPUTS_CORE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE)",
        inputs: &IFFTN_INPUTS_SIZE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, symflag)",
        inputs: &IFFTN_INPUTS_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE, symflag)",
        inputs: &IFFTN_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
];

const IFFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.ARG_COUNT",
    identifier: Some("RunMat:ifftn:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "ifftn: invalid argument count",
};

const IFFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SIZE",
    identifier: Some("RunMat:ifftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "ifftn: invalid SIZE argument",
};

const IFFTN_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifftn:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifftn: invalid symmetry flag usage",
};

const IFFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_INPUT",
    identifier: Some("RunMat:ifftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifftn: invalid input",
};

const IFFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INTERNAL",
    identifier: Some("RunMat:ifftn:Internal"),
    when: "IFFTN execution or tensor shaping fails.",
    message: "ifftn: internal error",
};
const IFFTN_ERROR_PROVIDER_INTEGRITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.PROVIDER_INTEGRITY",
    identifier: Some("RunMat:ifftn:ProviderIntegrity"),
    when:
        "The provider returns ownership, shape, or physical metadata inconsistent with the request.",
    message: "ifftn: provider integrity error",
};

const IFFTN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFTN_ERROR_ARG_COUNT,
    IFFTN_ERROR_INVALID_SIZE,
    IFFTN_ERROR_INVALID_SYMFLAG,
    IFFTN_ERROR_INVALID_INPUT,
    IFFTN_ERROR_INTERNAL,
    IFFTN_ERROR_PROVIDER_INTEGRITY,
];

pub const IFFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFTN_ERRORS,
};

fn ifftn_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifftn_error_with_message(error.message, error)
}

fn ifftn_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifftn_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifftn_error_with_source(
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

fn ifftn_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ifftn_provider_error(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!(
        "ifftn: provider integrity error: {}",
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier(
        IFFTN_ERROR_PROVIDER_INTEGRITY
            .identifier
            .expect("ifftn provider-integrity descriptor identifier"),
    )
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

#[runtime_builtin(
    name = "ifftn",
    category = "math/fft",
    summary = "Compute the N-dimensional inverse discrete Fourier transform (IDFT) of numeric or complex data.",
    keywords = "ifftn,inverse nd fft,n-dimensional inverse fourier transform,gpu",
    type_resolver(ifftn_type),
    descriptor(crate::builtins::math::fft::ifftn::IFFTN_DESCRIPTOR),
    extensions(crate::builtins::math::fft::ifftn::IFFTN_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::ifftn::IFFTN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::ifftn"
)]
async fn ifftn_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "ifftn")?;
    if is_wide_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFTN_WIDE_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
        ensure_wide_integer_data_exact(&value, BUILTIN_NAME)?;
    }
    if rest.iter().any(is_wide_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFTN_WIDE_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (sizes, symmetric) = parse_ifftn_arguments(&rest)?;
    if let Some(ref sizes) = sizes {
        if sizes.len() < ifftn_input_rank(&value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &IFFTN_SHORT_SIZE_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if sizes.contains(&0) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &IFFTN_ZERO_SIZE_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    match value {
        Value::GpuTensor(handle) => ifftn_gpu(handle, sizes, symmetric).await,
        other => ifftn_host(other, sizes, symmetric),
    }
}

fn ifftn_host(value: Value, sizes: Option<Vec<usize>>, symmetric: bool) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        ifftn_error_with_source(
            &IFFTN_ERROR_INVALID_INPUT,
            "input conversion failed",
            source,
        )
    })?;
    let transformed = ifftn_complex_tensor(tensor, sizes)?;
    finalize_ifftn_output(transformed, symmetric)
}

async fn ifftn_gpu(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    if let Some(ref spec) = sizes {
        if spec.is_empty() || spec.contains(&0) {
            return ifftn_gpu_fallback(handle, sizes, symmetric).await;
        }
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if runmat_accelerate_api::handle_integer_type(&handle).is_some()
            || runmat_accelerate_api::handle_is_logical(&handle)
        {
            return ifftn_gpu_fallback(handle, sizes, symmetric).await;
        }
        let mut current = handle.clone();
        let mut logical_shape = current.shape.clone();
        if logical_shape.is_empty() {
            logical_shape.push(1);
        }
        let axis_count = sizes
            .as_ref()
            .map(|v| v.len())
            .unwrap_or_else(|| logical_shape.len());
        let precision = runmat_accelerate_api::handle_precision(&handle)
            .unwrap_or_else(|| provider.precision());
        let input_metadata = gpu_metadata_snapshot(&handle);

        for axis in 0..axis_count {
            let len = sizes.as_ref().and_then(|v| v.get(axis).copied());
            while logical_shape.len() <= axis {
                logical_shape.push(1);
            }
            if let Some(len) = len {
                logical_shape[axis] = len;
            }
            match provider.ifft_dim(&current, len, axis).await {
                Ok(next) => {
                    if same_gpu_handle(&next, &handle) || same_gpu_handle(&next, &current) {
                        if same_gpu_handle(&next, &handle) {
                            restore_gpu_metadata(&handle, input_metadata);
                        }
                        free_rejected_provider_fft_output(provider, &next, &[&handle, &current]);
                        if !same_gpu_handle(&current, &handle) {
                            free_rejected_provider_fft_output(provider, &current, &[&handle]);
                        }
                        return Err(ifftn_provider_error(format!(
                            "ifft_dim pass {} aliased a protected handle",
                            axis + 1
                        )));
                    }
                    if !valid_provider_fft_output(
                        provider,
                        &next,
                        &logical_shape,
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                        precision,
                    ) {
                        free_rejected_provider_fft_output(provider, &next, &[&handle, &current]);
                        if !same_gpu_handle(&current, &handle) {
                            free_rejected_provider_fft_output(provider, &current, &[&handle]);
                        }
                        return Err(ifftn_provider_error(format!(
                            "ifft_dim pass {} returned malformed metadata",
                            axis + 1
                        )));
                    }
                    if !same_gpu_handle(&current, &handle) {
                        free_rejected_provider_fft_output(provider, &current, &[&handle, &next]);
                    }
                    current = next;
                }
                Err(error) if provider_operation_unsupported(&error, "ifft_dim") => {
                    let downloaded = if same_gpu_handle(&current, &handle) {
                        gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await
                    } else {
                        let result = download_provider_complex_tensor(
                            provider,
                            &current,
                            BUILTIN_NAME,
                            false,
                        )
                        .await;
                        free_rejected_provider_fft_output(provider, &current, &[&handle]);
                        result
                    }
                    .map_err(|error| {
                        ifftn_provider_error(format!(
                            "partial provider result download failed: {error}"
                        ))
                    })?;
                    let mut transformed = downloaded;
                    for remaining_axis in axis..axis_count {
                        let remaining_len = sizes
                            .as_ref()
                            .and_then(|values| values.get(remaining_axis).copied());
                        transformed = transform_complex_tensor(
                            transformed,
                            remaining_len,
                            Some(remaining_axis + 1),
                            TransformDirection::Inverse,
                            BUILTIN_NAME,
                        )?;
                    }
                    return restore_ifftn_gpu_result(&handle, transformed, symmetric);
                }
                Err(error) => {
                    if !same_gpu_handle(&current, &handle) {
                        free_rejected_provider_fft_output(provider, &current, &[&handle]);
                    }
                    return Err(ifftn_provider_error(format!(
                        "ifft_dim pass {} failed: {error}",
                        axis + 1
                    )));
                }
            }
        }

        if !symmetric {
            return Ok(Value::GpuTensor(current));
        }
        let complex = download_provider_complex_tensor(provider, &current, BUILTIN_NAME, true)
            .await
            .map_err(|error| {
                ifftn_provider_error(format!("provider result download failed: {error}"))
            })?;
        return restore_ifftn_gpu_result(&handle, complex, true);
    }

    ifftn_gpu_fallback(handle, sizes, symmetric).await
}

async fn ifftn_gpu_fallback(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            ifftn_error_with_source(&IFFTN_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = ifftn_complex_tensor(complex, sizes)?;
    restore_ifftn_gpu_result(&handle, transformed, symmetric)
}

fn restore_ifftn_gpu_result(
    source: &GpuTensorHandle,
    transformed: ComplexTensor,
    symmetric: bool,
) -> BuiltinResult<Value> {
    if symmetric {
        let Value::Tensor(real) = finalize_ifftn_output(transformed, true)? else {
            unreachable!("symmetric ifftn produces a real tensor")
        };
        restore_real_gpu_result(source, &real, BUILTIN_NAME)
    } else {
        restore_complex_gpu_result(source, &transformed, BUILTIN_NAME)
    }
}

fn ifftn_complex_tensor(
    tensor: ComplexTensor,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<ComplexTensor> {
    transform_nd_complex_tensor(
        tensor,
        sizes.as_deref(),
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| ifftn_error_with_source(&IFFTN_ERROR_INTERNAL, "transform failed", source))
}

fn finalize_ifftn_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME).map_err(|source| {
            ifftn_error_with_source(
                &IFFTN_ERROR_INTERNAL,
                "real-value extraction failed",
                source,
            )
        })
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

fn parse_ifftn_arguments(args: &[Value]) -> BuiltinResult<(Option<Vec<usize>>, bool)> {
    if args.is_empty() {
        return Ok((None, false));
    }

    let (symflag, rem) = split_symflag(args)?;
    let symmetric = symflag.unwrap_or(false);

    let sizes = match rem.len() {
        0 => None,
        1 => Some(parse_sizes_value(&rem[0])?),
        _ => return Err(ifftn_error(&IFFTN_ERROR_ARG_COUNT)),
    };
    Ok((sizes, symmetric))
}

fn split_symflag(args: &[Value]) -> BuiltinResult<(Option<bool>, &[Value])> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last, BUILTIN_NAME).map_err(|source| {
            ifftn_error_with_source(&IFFTN_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })? {
            for value in rest {
                if parse_symflag(value, BUILTIN_NAME)
                    .map_err(|source| {
                        ifftn_error_with_source(
                            &IFFTN_ERROR_INVALID_SYMFLAG,
                            "symflag parse failed",
                            source,
                        )
                    })?
                    .is_some()
                {
                    return Err(ifftn_error_with_detail(
                        &IFFTN_ERROR_INVALID_SYMFLAG,
                        "symmetry flag must appear once at the end",
                    ));
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    for value in args {
        if parse_symflag(value, BUILTIN_NAME)
            .map_err(|source| {
                ifftn_error_with_source(
                    &IFFTN_ERROR_INVALID_SYMFLAG,
                    "symflag parse failed",
                    source,
                )
            })?
            .is_some()
        {
            return Err(ifftn_error_with_detail(
                &IFFTN_ERROR_INVALID_SYMFLAG,
                "symmetry flag must appear as the final argument",
            ));
        }
    }

    Ok((None, args))
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    validate_size_vector_shape(value)?;
    parse_nd_sizes_value(value, BUILTIN_NAME).map_err(|source| {
        ifftn_error_with_detail(
            &IFFTN_ERROR_INVALID_SIZE,
            format!("SIZE parse failed: {source}"),
        )
    })
}

fn ifftn_input_rank(value: &Value) -> usize {
    let rank = match value {
        Value::Tensor(tensor) => tensor.shape.len(),
        Value::ComplexTensor(tensor) => tensor.shape.len(),
        Value::LogicalArray(array) => array.shape.len(),
        Value::GpuTensor(handle) => handle.shape.len(),
        _ => 2,
    };
    rank.max(2)
}

fn validate_size_vector_shape(value: &Value) -> BuiltinResult<()> {
    let vector_shape = |shape: &[usize]| {
        shape.is_empty()
            || shape.len() == 1
            || (shape.len() == 2 && (shape[0] == 1 || shape[1] == 1))
    };
    match value {
        Value::Tensor(tensor) if !vector_shape(&tensor.shape) => Err(ifftn_error_with_detail(
            &IFFTN_ERROR_INVALID_SIZE,
            "SIZE must be a row or column vector",
        )),
        Value::LogicalArray(array) if !vector_shape(&array.shape) => Err(ifftn_error_with_detail(
            &IFFTN_ERROR_INVALID_SIZE,
            "SIZE must be a row or column vector",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(ifftn_error_with_detail(
            &IFFTN_ERROR_INVALID_SIZE,
            "SIZE must be real-valued",
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::Tensor;

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn ifftn_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifftn builtin");
        let descriptor = builtin.descriptor.expect("ifftn descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifftn(X)"));
        assert!(labels.contains(&"Y = ifftn(X, SIZE)"));
        assert!(labels.contains(&"Y = ifftn(X, symflag)"));
        assert!(labels.contains(&"Y = ifftn(X, SIZE, symflag)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFTN.INVALID_SYMFLAG"));
    }

    #[test]
    fn ifftn_roundtrip_matches_input_real_part() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input.clone()), BUILTIN_NAME).unwrap();
        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let freq = fft_complex_tensor(b, None, Some(3)).unwrap();
        let back = ifftn_complex_tensor(freq, None).unwrap();
        assert_eq!(back.shape, vec![2, 2, 2]);
        for (idx, (re, im)) in back.materialize_f64().iter().enumerate() {
            assert!((*re - input.materialize_f64()[idx]).abs() < 1e-10);
            assert!(im.abs() < 1e-10);
        }
    }

    #[test]
    fn ifftn_accepts_symmetric_flag() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input.clone()), BUILTIN_NAME).unwrap();
        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let freq = fft_complex_tensor(b, None, Some(3)).unwrap();

        let result = block_on(ifftn_builtin(
            Value::ComplexTensor(freq),
            vec![Value::from("symmetric")],
        ))
        .expect("ifftn symmetric");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                for (got, expected) in t
                    .materialize_f64()
                    .iter()
                    .zip(input.materialize_f64().iter())
                {
                    assert!((*got - *expected).abs() < 1e-10);
                }
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifftn_requires_symflag_final_position() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let size = Tensor::new(vec![2.0, 2.0, 2.0], vec![1, 3]).unwrap();
        let err = block_on(ifftn_builtin(
            Value::Tensor(input),
            vec![Value::from("symmetric"), Value::Tensor(size)],
        ))
        .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFTN_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFTN_ERROR_INVALID_SYMFLAG.message));
    }

    #[test]
    fn ifftn_rejects_invalid_argument_count() {
        let err = parse_ifftn_arguments(&[
            Value::Num(2.0),
            Value::Num(2.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ])
        .unwrap_err();
        assert_eq!(error_identifier(&err), IFFTN_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(IFFTN_ERROR_ARG_COUNT.message));
    }

    #[test]
    fn ifftn_accepts_documented_logical_size_and_rejects_complex_size() {
        let (size, symmetric) = parse_ifftn_arguments(&[Value::Bool(true)]).unwrap();
        assert_eq!(size, Some(vec![1]));
        assert!(!symmetric);

        let err = parse_ifftn_arguments(&[Value::Complex(1.0, 0.0)]).unwrap_err();
        assert_eq!(error_identifier(&err), IFFTN_ERROR_INVALID_SIZE.identifier);
        assert!(error_message(err).contains(IFFTN_ERROR_INVALID_SIZE.message));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ifftn_wgpu_matches_cpu_and_preserves_source_residency() {
        let _guard = crate::builtins::common::test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        let input =
            Tensor::new((1..=8).map(|value| value as f64).collect(), vec![2, 2, 2]).expect("input");
        let complex = value_to_complex_tensor(Value::Tensor(input), BUILTIN_NAME).unwrap();
        let frequency =
            transform_nd_complex_tensor(complex, None, TransformDirection::Forward, "fftn")
                .expect("frequency data");
        let interleaved = frequency
            .materialize_f64()
            .iter()
            .flat_map(|(real, imag)| [*real, *imag])
            .collect::<Vec<_>>();
        let raw = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &interleaved,
                shape: &[2, 2, 2, 2],
            })
            .expect("upload spectrum");
        let source = GpuTensorHandle {
            shape: frequency.shape.clone(),
            device_id: raw.device_id,
            buffer_id: raw.buffer_id,
            descriptor: runmat_accelerate_api::GpuTensorDescriptor {
                storage: Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved),
                ..raw.descriptor
            },
        };
        let gpu = block_on(ifftn_builtin(Value::GpuTensor(source.clone()), Vec::new()))
            .expect("resident ifftn");
        let cpu = block_on(ifftn_builtin(Value::ComplexTensor(frequency), Vec::new()))
            .expect("host ifftn");
        let gpu = block_on(crate::dispatcher::gather_if_needed_async(&gpu)).expect("gather result");
        let cpu = value_to_complex_tensor(cpu, BUILTIN_NAME).expect("host complex result");
        let gpu = value_to_complex_tensor(gpu, BUILTIN_NAME).expect("gathered complex result");
        assert_eq!(gpu.shape, cpu.shape);
        let tolerance = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-4,
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
        };
        for (actual, expected) in gpu
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert!((actual.0 - expected.0).abs() < tolerance);
            assert!((actual.1 - expected.1).abs() < tolerance);
        }
        assert!(runmat_accelerate_api::provider_for_handle(&source).is_some());
        provider.free(&source).ok();
        runmat_accelerate_api::clear_residency(&source);
    }
}
