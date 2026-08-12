//! MATLAB-compatible `ifft` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, default_dimension, download_provider_complex_tensor,
    ensure_wide_integer_data_exact, free_rejected_provider_fft_output, gather_gpu_complex_tensor,
    gpu_metadata_snapshot, is_wide_integer_value, parse_length, parse_symflag,
    provider_operation_unsupported, restore_complex_gpu_result, restore_gpu_metadata,
    restore_real_gpu_result, same_gpu_handle, transform_complex_tensor, valid_provider_fft_output,
    value_to_complex_tensor, TransformDirection,
};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use crate::builtins::math::fft::type_resolvers::ifft_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifft",
    op_kind: GpuOpKind::Custom("ifft"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers should expose `ifft_dim` (or reuse `fft_dim` with inverse scaling); when absent, the runtime gathers to the host and evaluates the inverse FFT in software.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifft",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Inverse FFT boundaries are not currently fused; fusion plans terminate before invoking `ifft`.",
};

const BUILTIN_NAME: &str = "ifft";

const IFFT_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftWideIntegerDataExtension"),
};

const IFFT_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft with int64 or uint64 N or DIM controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IfftWideIntegerControlExtension"),
};

pub const IFFT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [IFFT_WIDE_DATA_EXTENSION, IFFT_WIDE_CONTROL_EXTENSION];

const IFFT_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "Documented integer data enters the floating inverse-FFT domain and produces double output.",
}];

const IFFT_WIDE_INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[runmat_builtins::BuiltinIntegerClass::Int64, runmat_builtins::BuiltinIntegerClass::Uint64],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Host wide integers are gated and must be exactly representable as double; resident wide integers reject before gather.",
}];

const IFFT_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N or DIM",
    classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "N and DIM are parsed as exact structural controls.",
}];

const IFFT_WIDE_INTEGER_CONTROLS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N or DIM",
    classes: &[runmat_builtins::BuiltinIntegerClass::Int64, runmat_builtins::BuiltinIntegerClass::Uint64],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "Wide structural controls are independently gated and decoded from authoritative integer storage.",
}];

pub const IFFT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft(integer_X, ...)", inputs: &IFFT_INTEGER_DATA, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer data crosses once into double; symmetric output is real and nonsymmetric output is complex." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft(X, integer_N, integer_DIM, ...)", inputs: &IFFT_INTEGER_CONTROLS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Documented controls are exact and are classified before provider access." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft(int64_or_uint64_X, ...)", inputs: &IFFT_WIDE_INTEGER_DATA, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only wide data cannot silently round at the double boundary." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft(X, int64_or_uint64_N_or_DIM, ...)", inputs: &IFFT_WIDE_INTEGER_CONTROLS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only wide controls are independently gated and parsed exactly." },
];

const IFFT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Inverse FFT result.",
}];

const IFFT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

const IFFT_INPUTS_WITH_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
];

const IFFT_INPUTS_WITH_SYMFLAG: [BuiltinParamDescriptor; 2] = [
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

const IFFT_INPUTS_WITH_N_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("first non-singleton dimension"),
        description: "Dimension to transform along.",
    },
];

const IFFT_INPUTS_WITH_N_SYMFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT_INPUTS_WITH_N_DIM_SYMFLAG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform length along selected dimension.",
    },
    BuiltinParamDescriptor {
        name: "DIM",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("first non-singleton dimension"),
        description: "Dimension to transform along.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X)",
        inputs: &IFFT_INPUTS_CORE,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N)",
        inputs: &IFFT_INPUTS_WITH_N,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, symflag)",
        inputs: &IFFT_INPUTS_WITH_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, DIM)",
        inputs: &IFFT_INPUTS_WITH_N_DIM,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, symflag)",
        inputs: &IFFT_INPUTS_WITH_N_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft(X, N, DIM, symflag)",
        inputs: &IFFT_INPUTS_WITH_N_DIM_SYMFLAG,
        outputs: &IFFT_OUTPUT,
    },
];

const IFFT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.ARG_COUNT",
    identifier: Some("RunMat:ifft:ArgCount"),
    when: "More than four input arguments are supplied.",
    message: "ifft: invalid argument count",
};

const IFFT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_LENGTH",
    identifier: Some("RunMat:ifft:InvalidLength"),
    when: "Length argument N is invalid.",
    message: "ifft: invalid length argument",
};

const IFFT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_DIMENSION",
    identifier: Some("RunMat:ifft:InvalidDimension"),
    when: "Dimension argument DIM is invalid.",
    message: "ifft: invalid dimension argument",
};

const IFFT_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifft:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifft: invalid symmetry flag usage",
};

const IFFT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INVALID_INPUT",
    identifier: Some("RunMat:ifft:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifft: invalid input",
};

const IFFT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT.INTERNAL",
    identifier: Some("RunMat:ifft:Internal"),
    when: "IFFT execution or tensor shaping fails.",
    message: "ifft: internal error",
};

const IFFT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFT_ERROR_ARG_COUNT,
    IFFT_ERROR_INVALID_LENGTH,
    IFFT_ERROR_INVALID_DIMENSION,
    IFFT_ERROR_INVALID_SYMFLAG,
    IFFT_ERROR_INVALID_INPUT,
    IFFT_ERROR_INTERNAL,
];

pub const IFFT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFT_ERRORS,
};

fn ifft_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifft_error_with_message(error.message, error)
}

fn ifft_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifft_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifft_error_with_source(
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

fn ifft_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ifft_provider_error(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!(
        "ifft: provider integrity error: {}",
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier("RunMat:ifft:ProviderIntegrity")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

#[runtime_builtin(
    name = "ifft",
    category = "math/fft",
    summary = "Compute inverse discrete Fourier transforms.",
    keywords = "ifft,inverse fft,inverse fourier transform,symmetric,gpu",
    type_resolver(ifft_type),
    descriptor(crate::builtins::math::fft::ifft::IFFT_DESCRIPTOR),
    extensions(crate::builtins::math::fft::ifft::IFFT_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::ifft::IFFT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::ifft"
)]
async fn ifft_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "ifft")?;
    if is_wide_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT_WIDE_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
        ensure_wide_integer_data_exact(&value, BUILTIN_NAME)?;
    }
    if rest.iter().any(is_wide_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT_WIDE_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let (length, dimension, symmetric) = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => ifft_gpu(handle, length, dimension, symmetric).await,
        other => ifft_host(other, length, dimension, symmetric),
    }
}

fn ifft_host(
    value: Value,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        ifft_error_with_source(&IFFT_ERROR_INVALID_INPUT, "input conversion failed", source)
    })?;
    let transformed = ifft_complex_tensor(tensor, length, dimension)?;
    finalize_ifft_output(transformed, symmetric)
}

async fn ifft_gpu(
    handle: GpuTensorHandle,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let mut logical_shape = normalize_scalar_shape(&handle.shape);

    let dim_one_based = match dimension {
        Some(0) => return Err(ifft_error(&IFFT_ERROR_INVALID_DIMENSION)),
        Some(dim) => dim,
        None => default_dimension(&logical_shape),
    };
    let dim_index = dim_one_based - 1;

    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }

    let current_len = logical_shape.get(dim_index).copied().unwrap_or(0);
    let target_len = length.unwrap_or(current_len);

    let expected_shape = {
        let mut shape = logical_shape.clone();
        shape[dim_index] = target_len;
        shape
    };
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let input_is_floating = runmat_accelerate_api::handle_integer_type(&handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(&handle);
        let precision = runmat_accelerate_api::handle_precision(&handle)
            .unwrap_or_else(|| provider.precision());
        if target_len != 0 && input_is_floating {
            let input_metadata = gpu_metadata_snapshot(&handle);
            match provider.ifft_dim(&handle, length, dim_index).await {
                Ok(out) => {
                    if same_gpu_handle(&handle, &out) {
                        restore_gpu_metadata(&handle, input_metadata);
                        return Err(ifft_provider_error("ifft_dim aliased its input"));
                    }
                    if !valid_provider_fft_output(
                        provider,
                        &out,
                        &expected_shape,
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                        precision,
                    ) {
                        free_rejected_provider_fft_output(provider, &out, &[&handle]);
                        return Err(ifft_provider_error("ifft_dim returned malformed metadata"));
                    }
                    if !symmetric {
                        return Ok(Value::GpuTensor(out));
                    }
                    let complex =
                        match download_provider_complex_tensor(provider, &out, BUILTIN_NAME, true)
                            .await
                        {
                            Ok(complex) => complex,
                            Err(error) => {
                                return Err(ifft_provider_error(format!(
                                    "provider result download failed: {error}"
                                )));
                            }
                        };
                    let Value::Tensor(real) = finalize_ifft_output(complex, true)? else {
                        unreachable!("symmetric ifft produces a real tensor")
                    };
                    return restore_real_gpu_result(&handle, &real, BUILTIN_NAME);
                }
                Err(error) if provider_operation_unsupported(&error, "ifft_dim") => {}
                Err(error) => return Err(ifft_provider_error(format!("ifft_dim failed: {error}"))),
            }
        }
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            ifft_error_with_source(&IFFT_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = ifft_complex_tensor(complex, length, dimension)?;
    if symmetric {
        let Value::Tensor(real) = finalize_ifft_output(transformed, true)? else {
            unreachable!("symmetric ifft produces a real tensor")
        };
        restore_real_gpu_result(&handle, &real, BUILTIN_NAME)
    } else {
        restore_complex_gpu_result(&handle, &transformed, BUILTIN_NAME)
    }
}

pub(super) fn ifft_complex_tensor(
    tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    transform_complex_tensor(
        tensor,
        length,
        dimension,
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| ifft_error_with_source(&IFFT_ERROR_INTERNAL, "transform failed", source))
}

fn finalize_ifft_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME).map_err(|source| {
            ifft_error_with_source(&IFFT_ERROR_INTERNAL, "real-value extraction failed", source)
        })
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
        .await
        .map_err(|detail| ifft_error_with_detail(&IFFT_ERROR_INVALID_DIMENSION, detail))?
        .ok_or_else(|| {
            ifft_error_with_detail(&IFFT_ERROR_INVALID_DIMENSION, format!("received {value:?}"))
        })
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>, bool)> {
    match args.len() {
        0 => Ok((None, None, false)),
        1 => match parse_symflag(&args[0], BUILTIN_NAME).map_err(|source| {
            ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })? {
            Some(flag) => Ok((None, None, flag)),
            None => {
                let len = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                    ifft_error_with_source(
                        &IFFT_ERROR_INVALID_LENGTH,
                        "length parse failed",
                        source,
                    )
                })?;
                Ok((len, None, false))
            }
        },
        2 => {
            let first_flag = parse_symflag(&args[0], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
            })?;
            let second_flag = parse_symflag(&args[1], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
            })?;
            if let Some(flag) = second_flag {
                if first_flag.is_some() {
                    return Err(ifft_error_with_detail(
                        &IFFT_ERROR_INVALID_SYMFLAG,
                        "symmetry flag must appear as the final argument",
                    ));
                }
                let len = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                    ifft_error_with_source(
                        &IFFT_ERROR_INVALID_LENGTH,
                        "length parse failed",
                        source,
                    )
                })?;
                Ok((len, None, flag))
            } else if first_flag.is_some() {
                Err(ifft_error_with_detail(
                    &IFFT_ERROR_INVALID_SYMFLAG,
                    "symmetry flag must appear as the final argument",
                ))
            } else {
                let len = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                    ifft_error_with_source(
                        &IFFT_ERROR_INVALID_LENGTH,
                        "length parse failed",
                        source,
                    )
                })?;
                let dim = Some(parse_dimension_arg(&args[1]).await?);
                Ok((len, dim, false))
            }
        }
        3 => {
            let first_flag = parse_symflag(&args[0], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
            })?;
            let second_flag = parse_symflag(&args[1], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
            })?;
            let third_flag = parse_symflag(&args[2], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
            })?;
            let symmetry = third_flag.ok_or_else(|| {
                ifft_error_with_detail(
                    &IFFT_ERROR_INVALID_SYMFLAG,
                    "expected 'symmetric' or 'nonsymmetric' as the final argument",
                )
            })?;
            if first_flag.is_some() || second_flag.is_some() {
                return Err(ifft_error_with_detail(
                    &IFFT_ERROR_INVALID_SYMFLAG,
                    "symmetry flag may only appear once at the end",
                ));
            }
            let len = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                ifft_error_with_source(&IFFT_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            let dim = Some(parse_dimension_arg(&args[1]).await?);
            Ok((len, dim, symmetry))
        }
        _ => Err(ifft_error(&IFFT_ERROR_ARG_COUNT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::math::fft::common;
    use futures::executor::block_on;
    use num_complex::Complex;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type};
    use runmat_value::{ComplexTensor as HostComplexTensor, IntValue, Tensor};
    #[cfg(feature = "wgpu")]
    use runmat_value::{IntegerStorage, LogicalArray};
    use rustfft::FftPlanner;

    fn approx_eq((a_re, a_im): (f64, f64), (b_re, b_im): (f64, f64), tol: f64) -> bool {
        (a_re - b_re).abs() <= tol && (a_im - b_im).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    fn value_as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(t) => t,
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu ifft output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            Value::Tensor(t) => HostComplexTensor::new(
                t.materialize_f64()
                    .into_iter()
                    .map(|re| (re, 0.0))
                    .collect(),
                t.shape,
            )
            .unwrap(),
            Value::Num(n) => HostComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).unwrap(),
            Value::Int(i) => HostComplexTensor::new(vec![(i.to_f64(), 0.0)], vec![1, 1]).unwrap(),
            other => panic!("unexpected value kind {other:?}"),
        }
    }

    #[test]
    fn ifft_type_preserves_shape() {
        let out = ifft_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(2)])
            }
        );
    }

    #[test]
    fn ifft_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifft builtin");
        let descriptor = builtin.descriptor.expect("ifft descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifft(X)"));
        assert!(labels.contains(&"Y = ifft(X, N)"));
        assert!(labels.contains(&"Y = ifft(X, symflag)"));
        assert!(labels.contains(&"Y = ifft(X, N, DIM)"));
        assert!(labels.contains(&"Y = ifft(X, N, symflag)"));
        assert!(labels.contains(&"Y = ifft(X, N, DIM, symflag)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFT.INVALID_SYMFLAG"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_inverts_known_fft() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), None, None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                let expected = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
                for (idx, actual) in ct.materialize_f64().iter().enumerate() {
                    assert!(approx_eq(*actual, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symmetric_returns_real_tensor() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, true).expect("ifft symmetric");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_length_returns_empty_tensor() {
        let spectrum = HostComplexTensor::new(Vec::new(), vec![0]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(0), None, false)
            .expect("ifft zero length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![0]);
                assert!(ct.materialize_f64().is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_dimension_argument_recovers_matrix() {
        let original = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mut spectrum = Vec::with_capacity(original.materialize_f64().len());
        let rows = original.shape[0];
        let cols = original.shape[1];
        for c in 0..cols {
            let mut column = Vec::with_capacity(rows);
            for r in 0..rows {
                column.push(Complex::new(original.materialize_f64()[r + c * rows], 0.0));
            }
            let mut fft = column.clone();
            FftPlanner::<f64>::new()
                .plan_fft_forward(rows)
                .process(&mut fft);
            for value in fft {
                spectrum.push((value.re, value.im));
            }
        }
        let freq = HostComplexTensor::new(spectrum, vec![2, 3]).unwrap();
        let result = ifft_host(Value::ComplexTensor(freq), None, Some(1), false).expect("ifft dim");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 3]);
                for (idx, (re, im)) in ct.materialize_f64().iter().enumerate() {
                    assert!(approx_eq(
                        (*re, *im),
                        (original.materialize_f64()[idx], 0.0),
                        1e-12
                    ));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_dimension_zero() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::Num(4.0),
                Value::Int(IntValue::I32(0)),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_accepts_scalar_tensor_dimension_argument() {
        let dim = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let (len, parsed_dim, symmetric) =
            block_on(parse_arguments(&[Value::Num(4.0), Value::Tensor(dim)]))
                .expect("parse arguments");
        assert_eq!(len, Some(4));
        assert_eq!(parsed_dim, Some(2));
        assert!(!symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_unknown_string_option() {
        let err = block_on(parse_arguments(&[Value::from("invalidflag")])).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFT_ERROR_INVALID_SYMFLAG.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_accepts_nonsymmetric_flag() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from("nonsymmetric")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(!symmetric);

        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, symmetric).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_requires_final_position() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::from("nonsymmetric"),
                Value::Num(4.0),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("symmetry flag must appear as the final argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_accepts_whitespace() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from(" symmetric ")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_padding_length_argument() {
        let spectrum = HostComplexTensor::new(vec![(4.0, 0.0)], vec![1]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(4), None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                for &(re, im) in &ct.materialize_f64() {
                    assert!((re - 1.0).abs() < 1e-12);
                    assert!(im.abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_truncates_when_length_is_smaller() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(2), None, false).expect("ifft");
        let mut expected = vec![Complex::new(10.0, 0.0), Complex::new(-2.0, 2.0)];
        FftPlanner::<f64>::new()
            .plan_fft_inverse(2)
            .process(&mut expected);
        for value in &mut expected {
            *value /= 2.0;
        }
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2]);
                for ((re, im), expected) in ct.materialize_f64().iter().zip(expected.iter()) {
                    assert!(approx_eq((*re, *im), (expected.re, expected.im), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_empty_length_with_symmetric_flag() {
        let empty = Tensor::new(Vec::new(), vec![0]).unwrap();
        let (len, dim, symmetric) = block_on(parse_arguments(&[
            Value::Tensor(empty),
            Value::from("symmetric"),
        ]))
        .expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[test]
    fn ifft_declares_documented_and_wide_integer_forms() {
        assert_eq!(IFFT_INTEGER_CAPABILITIES.len(), 4);
        assert_eq!(IFFT_EXTENSIONS.len(), 2);
        assert_eq!(IFFT_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 6);
        assert_eq!(IFFT_INTEGER_CAPABILITIES[2].inputs[0].classes.len(), 2);
        assert_eq!(
            IFFT_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
        assert_eq!(
            IFFT_INTEGER_CAPABILITIES[2].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }

    #[test]
    fn ifft_wide_integer_data_is_mode_gated_and_exact() {
        let input = || {
            Value::Tensor(
                Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![1, 2]), vec![1, 2])
                    .unwrap(),
            )
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error =
            ifft_builtin(input(), Vec::new()).expect_err("strict mode must reject wide data");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:IfftWideIntegerDataExtension")
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let output = ifft_builtin(input(), Vec::new()).expect("exact wide data");
        assert!(matches!(
            output,
            Value::ComplexTensor(_) | Value::Complex(_, _)
        ));
        let inexact = Value::Tensor(
            Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
                vec![1, 1],
            )
            .unwrap(),
        );
        let error = ifft_builtin(inexact, Vec::new()).expect_err("inexact wide data must reject");
        assert!(error.message().contains("exactly representable"));
    }

    #[test]
    fn ifft_documented_integer_data_returns_double_complex_storage() {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::I32(vec![1, 2, 3, 4]),
            vec![1, 4],
        )
        .unwrap();
        let output = ifft_builtin(Value::Tensor(input), Vec::new()).expect("integer ifft");
        let complex = value_as_complex_tensor(output);
        assert_eq!(complex.numeric_dtype(), runmat_value::NumericDType::F64);
    }

    #[test]
    fn ifft_wide_integer_controls_have_an_independent_gate() {
        let input = || Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap());
        let length = || Value::Int(IntValue::U64(2));
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = ifft_builtin(input(), vec![length()])
            .expect_err("strict mode must reject wide controls");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:IfftWideIntegerControlExtension")
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        ifft_builtin(input(), vec![length()]).expect("RunMat mode accepts exact wide controls");
    }

    #[test]
    fn ifft_preserves_single_for_complex_and_symmetric_outputs() {
        let spectrum = || {
            Value::ComplexTensor(
                HostComplexTensor::from_complex_storage(
                    runmat_value::ComplexStorage::F32(vec![(1.0, 0.0), (0.0, 0.0)]),
                    vec![1, 2],
                )
                .unwrap(),
            )
        };
        let complex = value_as_complex_tensor(
            ifft_builtin(spectrum(), Vec::new()).expect("complex single ifft"),
        );
        assert_eq!(complex.numeric_dtype(), runmat_value::NumericDType::F32);

        let real = ifft_builtin(spectrum(), vec![Value::from("symmetric")])
            .expect("symmetric single ifft");
        let Value::Tensor(real) = real else {
            panic!("expected real tensor")
        };
        assert_eq!(real.numeric_dtype(), runmat_value::NumericDType::F32);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let spectrum_handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![4],
                device_id: handle.device_id,
                buffer_id: handle.buffer_id,
            };
            runmat_accelerate_api::set_handle_storage(
                &spectrum_handle,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            let gpu =
                ifft_builtin(Value::GpuTensor(spectrum_handle.clone()), Vec::new()).expect("ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu = ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct
                .materialize_f64()
                .iter()
                .zip(cpu_ct.materialize_f64().iter())
            {
                assert!(approx_eq(*a, *b, 1e-12));
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_gpu_symmetric_returns_resident_real_tensor() {
        test_support::with_test_provider(|provider| {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let spectrum_handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![4],
                device_id: handle.device_id,
                buffer_id: handle.buffer_id,
            };
            runmat_accelerate_api::set_handle_storage(
                &spectrum_handle,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            let gpu = ifft_builtin(
                Value::GpuTensor(spectrum_handle.clone()),
                vec![Value::from("symmetric")],
            )
            .expect("ifft symmetric");
            match gpu {
                Value::GpuTensor(_) | Value::Tensor(_) => {
                    let gathered = test_support::gather(gpu).expect("gather symmetric real");
                    assert_eq!(gathered.materialize_f64().len(), 4);
                    assert_eq!(gathered.shape.first().copied().unwrap_or(0), 4);
                    for (idx, value) in gathered.materialize_f64().iter().enumerate() {
                        assert!((*value - (idx as f64 + 1.0)).abs() < 1e-10);
                    }
                }
                other => panic!("expected real output tensor, got {other:?}"),
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft_wgpu_matches_cpu() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let spectrum_handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![4],
                device_id: handle.device_id,
                buffer_id: handle.buffer_id,
            };
            runmat_accelerate_api::set_handle_storage(
                &spectrum_handle,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            let gpu = ifft_builtin(Value::GpuTensor(spectrum_handle.clone()), Vec::new())
                .expect("gpu ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu =
                ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("cpu ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            let tol = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
                runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
            };
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct
                .materialize_f64()
                .iter()
                .zip(cpu_ct.materialize_f64().iter())
            {
                assert!(approx_eq(*a, *b, tol), "{a:?} vs {b:?}");
            }
            provider.free(&handle).ok();
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft_f32_owner_returns_host_double_for_integer_and_logical_fallback() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            if provider.precision() != runmat_accelerate_api::ProviderPrecision::F32 {
                return;
            }
            let integer = Tensor::new_integer(IntegerStorage::I32(vec![1, 0]), vec![1, 2])
                .expect("integer tensor");
            let integer_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &integer)
                    .expect("upload integer");
            runmat_accelerate_api::set_handle_integer_type(
                &integer_handle,
                runmat_accelerate_api::IntegerElementType::I32,
            );
            let integer_output = ifft_builtin(Value::GpuTensor(integer_handle.clone()), Vec::new())
                .expect("integer fallback");
            let Value::ComplexTensor(integer_output) = integer_output else {
                panic!("F32 owner must return host complex double for integer ifft")
            };
            assert_eq!(
                integer_output.numeric_dtype(),
                runmat_value::NumericDType::F64
            );
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&integer_handle),
                Some(runmat_accelerate_api::IntegerElementType::I32)
            );

            let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical array");
            let logical_tensor = crate::builtins::common::tensor::logical_to_tensor(&logical)
                .expect("logical tensor");
            let logical_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &logical_tensor)
                    .expect("upload logical");
            runmat_accelerate_api::set_handle_logical(&logical_handle, true);
            let logical_output = ifft_builtin(
                Value::GpuTensor(logical_handle.clone()),
                vec![Value::from("symmetric")],
            )
            .expect("logical symmetric fallback");
            let Value::Tensor(logical_output) = logical_output else {
                panic!("F32 owner must return host real double for logical ifft")
            };
            assert_eq!(
                logical_output.numeric_dtype(),
                runmat_value::NumericDType::F64
            );
            assert!(runmat_accelerate_api::handle_is_logical(&logical_handle));

            provider.free(&integer_handle).ok();
            provider.free(&logical_handle).ok();
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft_wgpu_symmetric_uses_safe_real_restoration() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let spectrum = [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let uploaded = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &spectrum,
                    shape: &[4, 2],
                })
                .expect("upload spectrum");
            let input = GpuTensorHandle {
                shape: vec![4],
                device_id: uploaded.device_id,
                buffer_id: uploaded.buffer_id,
            };
            runmat_accelerate_api::set_handle_storage(
                &input,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            runmat_accelerate_api::set_handle_precision(&input, provider.precision());
            let output = ifft_builtin(
                Value::GpuTensor(input.clone()),
                vec![Value::from("symmetric")],
            )
            .expect("symmetric wgpu ifft");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("matching F32 owner should receive a resident real result")
            };
            assert_ne!(
                (output_handle.device_id, output_handle.buffer_id),
                (input.device_id, input.buffer_id)
            );
            assert_eq!(
                runmat_accelerate_api::handle_storage(output_handle),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
            let gathered = test_support::gather(output).expect("gather symmetric output");
            for (actual, expected) in gathered.materialize_f64().iter().zip([1.0, 2.0, 3.0, 4.0]) {
                assert!((*actual - expected).abs() <= 1e-5, "{actual} vs {expected}");
            }
            provider.free(&uploaded).ok();
        }
    }

    fn ifft_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifft_builtin(value, rest))
    }
}
