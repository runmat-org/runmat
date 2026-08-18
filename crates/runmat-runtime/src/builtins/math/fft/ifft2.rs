//! MATLAB-compatible `ifft2` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, download_provider_complex_tensor, ensure_wide_integer_data_exact,
    free_rejected_provider_fft_output, gather_gpu_complex_tensor, gpu_metadata_snapshot,
    is_wide_integer_value, parse_2d_lengths_from_data, parse_2d_lengths_from_tensor, parse_length,
    parse_symflag, provider_operation_unsupported, restore_complex_gpu_result,
    restore_gpu_metadata, restore_real_gpu_result, same_gpu_handle, transform_axes_complex_tensor,
    valid_provider_fft_output, value_to_complex_tensor, TransformDirection,
};
use super::ifft::ifft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::fft::type_resolvers::ifft2_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifft2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifft2",
    op_kind: GpuOpKind::Custom("ifft2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Performs two sequential `ifft_dim` passes (dimensions 0 and 1); falls back to host execution when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifft2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifft2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "ifft2 terminates fusion plans; fused kernels are not generated for multi-dimensional inverse FFTs.",
};

const BUILTIN_NAME: &str = "ifft2";

const IFFT2_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2 with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2WideIntegerDataExtension"),
};
const IFFT2_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2 with int64 or uint64 transform-size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2WideIntegerControlExtension"),
};
const IFFT2_SIZE_FORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-size-form",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2(X, SIZE) scalar/vector shorthand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2SizeFormExtension"),
};
const IFFT2_EMPTY_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ifft2-empty-zero-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ifft2 empty or zero transform sizes are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Ifft2EmptyZeroSizeExtension"),
};
pub const IFFT2_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    IFFT2_WIDE_DATA_EXTENSION,
    IFFT2_WIDE_CONTROL_EXTENSION,
    IFFT2_SIZE_FORM_EXTENSION,
    IFFT2_EMPTY_ZERO_SIZE_EXTENSION,
];

const IFFT2_DOCUMENTED_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer arrays enter the double two-dimensional inverse-FFT domain.",
    }];
const IFFT2_DOCUMENTED_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M and N",
        classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "M and N are exact positive scalar structural controls; logical scalars are also documented.",
    }];
const IFFT2_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Host wide data is gated and must cross the double boundary exactly.",
}];
const IFFT2_WIDE_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M, N, or SIZE",
        classes: &[
            runmat_builtins::BuiltinIntegerClass::Int64,
            runmat_builtins::BuiltinIntegerClass::Uint64,
        ],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "Wide structural controls are independently gated and parsed from authoritative integer storage.",
    }];
const IFFT2_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "The scalar or two-element SIZE shorthand is independently gated.",
}];
pub const IFFT2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft2(integer_X, ...)", inputs: &IFFT2_DOCUMENTED_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer data produces double output; symmetric changes complexity only." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft2(X, integer_M, integer_N, ...)", inputs: &IFFT2_DOCUMENTED_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Documented controls are parsed exactly before provider execution." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft2(int64_or_uint64_X, ...)", inputs: &IFFT2_WIDE_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only wide host data may not silently round; resident wide data rejects before gather." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft2(X, int64_or_uint64_M_or_N_or_SIZE, ...)", inputs: &IFFT2_WIDE_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only wide controls are independently gated and decoded exactly." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = ifft2(X, integer_SIZE, ...)", inputs: &IFFT2_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only scalar/two-vector shorthand; empty and zero sizes have a further gate." },
];

const IFFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D inverse FFT output.",
}];

const IFFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

const IFFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
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
        description: "Scalar N or two-element [M N] size vector.",
    },
];

const IFFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
];

const IFFT2_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
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

const IFFT2_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
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
        description: "Scalar N or two-element [M N] size vector.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT2_INPUTS_M_N_SYMFLAG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT2_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X)",
        inputs: &IFFT2_INPUTS_CORE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE)",
        inputs: &IFFT2_INPUTS_SIZE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N)",
        inputs: &IFFT2_INPUTS_M_N,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, symflag)",
        inputs: &IFFT2_INPUTS_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE, symflag)",
        inputs: &IFFT2_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N, symflag)",
        inputs: &IFFT2_INPUTS_M_N_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
];

const IFFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.ARG_COUNT",
    identifier: Some("RunMat:ifft2:ArgCount"),
    when: "More than four input arguments are supplied.",
    message: "ifft2: invalid argument count",
};

const IFFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_LENGTH",
    identifier: Some("RunMat:ifft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "ifft2: invalid transform length argument",
};

const IFFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:ifft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "ifft2: invalid size vector argument",
};

const IFFT2_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifft2:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifft2: invalid symmetry flag usage",
};

const IFFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_INPUT",
    identifier: Some("RunMat:ifft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifft2: invalid input",
};

const IFFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INTERNAL",
    identifier: Some("RunMat:ifft2:Internal"),
    when: "IFFT2 execution or tensor shaping fails.",
    message: "ifft2: internal error",
};

const IFFT2_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFT2_ERROR_ARG_COUNT,
    IFFT2_ERROR_INVALID_LENGTH,
    IFFT2_ERROR_INVALID_SIZE_VECTOR,
    IFFT2_ERROR_INVALID_SYMFLAG,
    IFFT2_ERROR_INVALID_INPUT,
    IFFT2_ERROR_INTERNAL,
];

pub const IFFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFT2_ERRORS,
};

fn ifft2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifft2_error_with_message(error.message, error)
}

fn ifft2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifft2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifft2_error_with_source(
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

fn ifft2_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ifft2_provider_error(detail: impl AsRef<str>) -> RuntimeError {
    build_runtime_error(format!(
        "ifft2: provider integrity error: {}",
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_identifier("RunMat:ifft2:ProviderIntegrity")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

#[runtime_builtin(
    name = "ifft2",
    category = "math/fft",
    summary = "Compute two-dimensional inverse Fourier transforms.",
    keywords = "ifft2,inverse fft,image reconstruction,gpu",
    type_resolver(ifft2_type),
    descriptor(crate::builtins::math::fft::ifft2::IFFT2_DESCRIPTOR),
    extensions(crate::builtins::math::fft::ifft2::IFFT2_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::ifft2::IFFT2_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::ifft2"
)]
async fn ifft2_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "ifft2")?;
    if is_wide_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT2_WIDE_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
        ensure_wide_integer_data_exact(&value, BUILTIN_NAME)?;
    }
    if rest.iter().any(is_wide_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT2_WIDE_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let parsed_symflag = rest
        .last()
        .map(|value| parse_symflag(value, BUILTIN_NAME))
        .transpose()
        .map_err(|source| {
            ifft2_error_with_source(&IFFT2_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })?
        .flatten();
    let control_count = parsed_symflag.map_or(rest.len(), |_| rest.len() - 1);
    if control_count == 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT2_SIZE_FORM_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let ((len_rows, len_cols), symmetric) = parse_ifft2_arguments(&rest)?;
    if control_count != 0
        && (len_rows.is_none() || len_cols.is_none() || len_rows == Some(0) || len_cols == Some(0))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IFFT2_EMPTY_ZERO_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match value {
        Value::GpuTensor(handle) => ifft2_gpu(handle, (len_rows, len_cols), symmetric).await,
        other => ifft2_host(other, (len_rows, len_cols), symmetric),
    }
}

fn ifft2_host(
    value: Value,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        ifft2_error_with_source(
            &IFFT2_ERROR_INVALID_INPUT,
            "input conversion failed",
            source,
        )
    })?;
    let transformed = ifft2_complex_tensor(tensor, lengths)?;
    finalize_ifft2_output(transformed, symmetric)
}

async fn ifft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return ifft2_gpu_fallback(handle, lengths, symmetric).await;
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let input_is_floating = runmat_accelerate_api::handle_integer_type(&handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(&handle);
        if input_is_floating {
            let precision = runmat_accelerate_api::handle_precision(&handle)
                .unwrap_or_else(|| provider.precision());
            let mut first_shape = handle.shape.clone();
            if first_shape.is_empty() {
                first_shape.push(1);
            }
            if let Some(len) = lengths.0 {
                first_shape[0] = len;
            }
            let input_metadata = gpu_metadata_snapshot(&handle);
            match provider.ifft_dim(&handle, lengths.0, 0).await {
                Ok(first) => {
                    if same_gpu_handle(&handle, &first) {
                        restore_gpu_metadata(&handle, input_metadata);
                        return Err(ifft2_provider_error(
                            "first ifft_dim pass aliased its input",
                        ));
                    }
                    if !valid_provider_fft_output(
                        provider,
                        &first,
                        &first_shape,
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                        precision,
                    ) {
                        free_rejected_provider_fft_output(provider, &first, &[&handle]);
                        return Err(ifft2_provider_error(
                            "first ifft_dim pass returned malformed metadata",
                        ));
                    }
                    let mut second_shape = first_shape.clone();
                    while second_shape.len() < 2 {
                        second_shape.push(1);
                    }
                    if let Some(len) = lengths.1 {
                        second_shape[1] = len;
                    }
                    match provider.ifft_dim(&first, lengths.1, 1).await {
                        Ok(second) => {
                            if same_gpu_handle(&second, &handle) || same_gpu_handle(&second, &first)
                            {
                                if same_gpu_handle(&second, &handle) {
                                    restore_gpu_metadata(&handle, input_metadata);
                                }
                                if same_gpu_handle(&second, &first) {
                                    free_rejected_provider_fft_output(provider, &first, &[&handle]);
                                } else {
                                    free_rejected_provider_fft_output(
                                        provider,
                                        &second,
                                        &[&handle, &first],
                                    );
                                    free_rejected_provider_fft_output(provider, &first, &[&handle]);
                                }
                                return Err(ifft2_provider_error(
                                    "second ifft_dim pass aliased a protected handle",
                                ));
                            }
                            if !valid_provider_fft_output(
                                provider,
                                &second,
                                &second_shape,
                                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                                precision,
                            ) {
                                free_rejected_provider_fft_output(
                                    provider,
                                    &second,
                                    &[&handle, &first],
                                );
                                free_rejected_provider_fft_output(provider, &first, &[&handle]);
                                return Err(ifft2_provider_error(
                                    "second ifft_dim pass returned malformed metadata",
                                ));
                            }
                            free_rejected_provider_fft_output(
                                provider,
                                &first,
                                &[&handle, &second],
                            );
                            if !symmetric {
                                return Ok(Value::GpuTensor(second));
                            }
                            let complex = download_provider_complex_tensor(
                                provider,
                                &second,
                                BUILTIN_NAME,
                                true,
                            )
                            .await
                            .map_err(|error| {
                                ifft2_provider_error(format!(
                                    "provider result download failed: {error}"
                                ))
                            })?;
                            let Value::Tensor(real) = finalize_ifft2_output(complex, true)? else {
                                unreachable!("symmetric ifft2 produces a real tensor")
                            };
                            return restore_real_gpu_result(&handle, &real, BUILTIN_NAME);
                        }
                        Err(error) if provider_operation_unsupported(&error, "ifft_dim") => {
                            let downloaded = download_provider_complex_tensor(
                                provider,
                                &first,
                                BUILTIN_NAME,
                                false,
                            )
                            .await;
                            free_rejected_provider_fft_output(provider, &first, &[&handle]);
                            let completed = ifft_complex_tensor(
                                downloaded.map_err(|error| {
                                    ifft2_provider_error(format!(
                                        "partial provider result download failed: {error}"
                                    ))
                                })?,
                                lengths.1,
                                Some(2),
                            )?;
                            return restore_ifft2_gpu_result(&handle, completed, symmetric);
                        }
                        Err(error) => {
                            free_rejected_provider_fft_output(provider, &first, &[&handle]);
                            return Err(ifft2_provider_error(format!(
                                "second ifft_dim pass failed: {error}"
                            )));
                        }
                    }
                }
                Err(error) if provider_operation_unsupported(&error, "ifft_dim") => {}
                Err(error) => {
                    return Err(ifft2_provider_error(format!(
                        "first ifft_dim pass failed: {error}"
                    )));
                }
            }
        }
    }

    ifft2_gpu_fallback(handle, lengths, symmetric).await
}

async fn ifft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            ifft2_error_with_source(&IFFT2_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = ifft2_complex_tensor(complex, lengths)?;
    restore_ifft2_gpu_result(&handle, transformed, symmetric)
}

fn restore_ifft2_gpu_result(
    source: &GpuTensorHandle,
    transformed: ComplexTensor,
    symmetric: bool,
) -> BuiltinResult<Value> {
    if symmetric {
        let Value::Tensor(real) = finalize_ifft2_output(transformed, true)? else {
            unreachable!("symmetric ifft2 produces a real tensor")
        };
        restore_real_gpu_result(source, &real, BUILTIN_NAME)
    } else {
        restore_complex_gpu_result(source, &transformed, BUILTIN_NAME)
    }
}

fn ifft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<ComplexTensor> {
    let (len_rows, len_cols) = lengths;
    transform_axes_complex_tensor(
        tensor,
        &[len_rows, len_cols],
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| ifft2_error_with_source(&IFFT2_ERROR_INTERNAL, "transform failed", source))
}

fn finalize_ifft2_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME).map_err(|source| {
            ifft2_error_with_source(
                &IFFT2_ERROR_INTERNAL,
                "real-value extraction failed",
                source,
            )
        })
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

type LengthPair = (Option<usize>, Option<usize>);
type LengthsAndSymmetry = (LengthPair, bool);

fn parse_ifft2_arguments(args: &[Value]) -> BuiltinResult<LengthsAndSymmetry> {
    if args.is_empty() {
        return Ok(((None, None), false));
    }

    let (maybe_flag, rem) = split_symflag(args)?;
    let mut symmetry = false;
    if let Some(flag) = maybe_flag {
        symmetry = flag;
    }

    let lengths = match rem.len() {
        0 => (None, None),
        1 => parse_ifft2_single(&rem[0])?,
        2 => {
            let rows = parse_length(&rem[0], BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_LENGTH,
                    "row-length parse failed",
                    source,
                )
            })?;
            let cols = parse_length(&rem[1], BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_LENGTH,
                    "column-length parse failed",
                    source,
                )
            })?;
            (rows, cols)
        }
        _ => return Err(ifft2_error(&IFFT2_ERROR_ARG_COUNT)),
    };

    Ok((lengths, symmetry))
}

fn split_symflag(args: &[Value]) -> BuiltinResult<(Option<bool>, &[Value])> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last, BUILTIN_NAME).map_err(|source| {
            ifft2_error_with_source(&IFFT2_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })? {
            // Ensure no earlier argument is also a symmetry flag.
            for value in rest {
                if parse_symflag(value, BUILTIN_NAME)
                    .map_err(|source| {
                        ifft2_error_with_source(
                            &IFFT2_ERROR_INVALID_SYMFLAG,
                            "symflag parse failed",
                            source,
                        )
                    })?
                    .is_some()
                {
                    return Err(ifft2_error_with_detail(
                        &IFFT2_ERROR_INVALID_SYMFLAG,
                        "symmetry flag must appear once at the end",
                    ));
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    // Validate that no argument except the last is a symmetry flag.
    for value in args {
        if parse_symflag(value, BUILTIN_NAME)
            .map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_SYMFLAG,
                    "symflag parse failed",
                    source,
                )
            })?
            .is_some()
        {
            return Err(ifft2_error_with_detail(
                &IFFT2_ERROR_INVALID_SYMFLAG,
                "symmetry flag must appear as the final argument",
            ));
        }
    }

    Ok((None, args))
}

fn parse_ifft2_single(value: &Value) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match value {
        Value::Tensor(tensor) => {
            parse_2d_lengths_from_tensor(tensor, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_detail(
                    &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical).map_err(|source| {
                ifft2_error_with_detail(
                    &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("logical size-vector conversion failed: {source}"),
                )
            })?;
            parse_2d_lengths_from_data(&tensor::tensor_into_values_f64(tensor), BUILTIN_NAME)
                .map_err(|source| {
                    ifft2_error_with_detail(
                        &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                        format!("size vector parse failed: {source}"),
                    )
                })
        }
        Value::Num(_) | Value::Int(_) => {
            let len = parse_length(value, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(&IFFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH));
            }
            let scalar = Value::Num(*re);
            let len = parse_length(&scalar, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(&IFFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::ComplexTensor(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::GpuTensor(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::Bool(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::math::fft::common;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        builtin_function_by_name, IntValue, IntegerStorage, ResolveContext, Tensor as HostTensor,
        Type,
    };

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    fn fft2_of_tensor(tensor: &HostTensor) -> ComplexTensor {
        let complex = value_to_complex_tensor(Value::Tensor(tensor.clone()), "fft2").unwrap();
        let first = super::super::fft::fft_complex_tensor(complex, None, Some(1)).unwrap();
        super::super::fft::fft_complex_tensor(first, None, Some(2)).unwrap()
    }

    fn value_to_host_complex(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu ifft2 output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_type_pads_rank() {
        let out = ifft2_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn ifft2_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifft2 builtin");
        assert_eq!(builtin.integer_capabilities.len(), 5);
        assert_eq!(builtin.extensions.len(), 4);
        let descriptor = builtin.descriptor.expect("ifft2 descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifft2(X)"));
        assert!(labels.contains(&"Y = ifft2(X, SIZE)"));
        assert!(labels.contains(&"Y = ifft2(X, M, N)"));
        assert!(labels.contains(&"Y = ifft2(X, symflag)"));
        assert!(labels.contains(&"Y = ifft2(X, SIZE, symflag)"));
        assert!(labels.contains(&"Y = ifft2(X, M, N, symflag)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFT2.INVALID_SYMFLAG"));
    }

    #[test]
    fn ifft2_integer_contract_gates_runmat_only_forms() {
        let input = || {
            Value::Tensor(
                HostTensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, 4]), vec![2, 2]).unwrap(),
            )
        };
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = ifft2_builtin(input(), Vec::new()).expect_err("wide data gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Ifft2WideIntegerDataExtension")
        );
        let ordinary = Value::Tensor(HostTensor::new(vec![1.0; 4], vec![2, 2]).unwrap());
        let error = ifft2_builtin(
            ordinary,
            vec![Value::Tensor(
                HostTensor::new_integer(IntegerStorage::U16(vec![2, 2]), vec![1, 2]).unwrap(),
            )],
        )
        .expect_err("SIZE shorthand gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Ifft2SizeFormExtension")
        );
    }

    #[test]
    fn ifft2_documented_integer_and_single_output_classes() {
        let integer =
            HostTensor::new_integer(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
        let output = ifft2_builtin(Value::Tensor(integer), Vec::new()).unwrap();
        assert_eq!(
            value_to_complex_tensor(output, BUILTIN_NAME)
                .unwrap()
                .numeric_dtype(),
            runmat_builtins::NumericDType::F64
        );
        let single = HostTensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = ifft2_builtin(Value::Tensor(single), Vec::new()).unwrap();
        assert_eq!(
            value_to_complex_tensor(output, BUILTIN_NAME)
                .unwrap()
                .numeric_dtype(),
            runmat_builtins::NumericDType::F32
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_inverts_known_fft2() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value =
            ifft2_builtin(Value::ComplexTensor(spectrum.clone()), Vec::new()).expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                for (idx, (re, im)) in out.materialize_f64().iter().enumerate() {
                    assert!(approx_eq(
                        (*re, *im),
                        (tensor.materialize_f64()[idx], 0.0),
                        1e-12
                    ));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_symmetric_returns_real() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::from("symmetric")],
        )
        .expect("ifft2 symmetric");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_accepts_nonsymmetric_flag() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::from("nonsymmetric")],
        )
        .expect("ifft2 nonsymmetric");
        let result = value_to_complex_tensor(value, "ifft2").expect("complex output");
        assert_eq!(result.shape, tensor.shape);
        for (idx, (re, im)) in result.materialize_f64().iter().enumerate() {
            assert!(approx_eq(
                (*re, *im),
                (tensor.materialize_f64()[idx], 0.0),
                1e-12
            ));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_accepts_scalar_length() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = HostTensor::new((0..9).map(|v| v as f64).collect(), vec![3, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_accepts_size_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let size = HostTensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let value = ifft2_builtin(Value::ComplexTensor(spectrum), vec![Value::Tensor(size)])
            .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_size_vector_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let size = HostTensor::new_integer(IntegerStorage::U16(vec![4, 2]), vec![1, 2]).unwrap();

        let value = ifft2_builtin(Value::ComplexTensor(spectrum), vec![Value::Tensor(size)])
            .expect("ifft2");

        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_treats_empty_lengths_as_defaults() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let empty_rows = HostTensor::new(vec![], vec![0]).unwrap();
        let empty_cols = HostTensor::new(vec![], vec![0]).unwrap();
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::Tensor(empty_rows), Value::Tensor(empty_cols)],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                for (idx, (re, im)) in out.materialize_f64().iter().enumerate() {
                    assert!(approx_eq(
                        (*re, *im),
                        (tensor.materialize_f64()[idx], 0.0),
                        1e-12
                    ));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_accepts_documented_boolean_lengths_and_gates_boolean_size_shorthand() {
        let tensor = HostTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::Bool(true)],
        )
        .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            Some(IFFT2_SIZE_FORM_EXTENSION.error_identifier.unwrap())
        );
        drop(strict);
        let result = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Bool(true), Value::Bool(true)],
        )
        .expect("documented logical M and N");
        let result = value_to_complex_tensor(result, "ifft2").expect("complex scalar result");
        assert_eq!(result.shape, vec![1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_rejects_excess_arguments() {
        let tensor = HostTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap_err();
        assert_eq!(error_identifier(&err), IFFT2_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(IFFT2_ERROR_ARG_COUNT.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_zero_lengths_return_empty_result() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(0))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert!(out.materialize_f64().is_empty());
                assert_eq!(out.shape, vec![0, 0]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = HostTensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let spectrum = fft2_of_tensor(&tensor);
            let view = HostTensorView {
                data: &spectrum
                    .materialize_f64()
                    .iter()
                    .flat_map(|(re, im)| [*re, *im])
                    .collect::<Vec<_>>(),
                shape: &[2, 4, 2],
            };
            let raw = provider.upload(&view).expect("upload spectrum");
            let second = runmat_accelerate_api::GpuTensorHandle {
                shape: spectrum.shape.clone(),
                device_id: raw.device_id,
                buffer_id: raw.buffer_id,
                descriptor: runmat_accelerate_api::GpuTensorDescriptor {
                    storage: Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved),
                    ..raw.descriptor
                },
            };

            let gpu =
                ifft2_builtin(Value::GpuTensor(second.clone()), Vec::new()).expect("ifft2 gpu");
            let cpu = ifft2_builtin(Value::ComplexTensor(spectrum.clone()), Vec::new())
                .expect("ifft2 cpu");

            let g = value_to_host_complex(gpu);
            let c = value_to_host_complex(cpu);
            assert_eq!(g.shape, c.shape);
            for (lhs, rhs) in g.materialize_f64().iter().zip(c.materialize_f64().iter()) {
                assert!(approx_eq(*lhs, *rhs, 1e-10), "{lhs:?} vs {rhs:?}");
            }
            provider.free(&raw).ok();
            provider.free(&second).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_handles_row_and_column_lengths() {
        let tensor = HostTensor::new((0..12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(5)), Value::Int(IntValue::I32(2))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![5, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_rejects_unknown_symmetry_flag() {
        let err = parse_ifft2_arguments(&[Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFT2_ERROR_INVALID_SYMFLAG.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_strict_mode_reports_invalid_symmetry_before_size_extension() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = ifft2_builtin(Value::Num(1.0), vec![Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_SYMFLAG.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_requires_symflag_last() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::from("symmetric"), Value::Int(IntValue::I32(2))],
        )
        .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFT2_ERROR_INVALID_SYMFLAG.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft2_wgpu_matches_cpu() {
        let provider = match std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
        }) {
            Ok(Ok(Some(provider))) => provider,
            _ => return,
        };

        let tensor = HostTensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let host_real_imag = spectrum
            .materialize_f64()
            .iter()
            .flat_map(|(re, im)| [*re, *im])
            .collect::<Vec<_>>();
        let view = HostTensorView {
            data: &host_real_imag,
            shape: &[4, 4, 2],
        };
        let raw = provider.upload(&view).expect("upload spectrum");
        let second = runmat_accelerate_api::GpuTensorHandle {
            shape: spectrum.shape.clone(),
            device_id: raw.device_id,
            buffer_id: raw.buffer_id,
            descriptor: runmat_accelerate_api::GpuTensorDescriptor {
                storage: Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved),
                ..raw.descriptor
            },
        };

        let gpu_val =
            ifft2_builtin(Value::GpuTensor(second.clone()), Vec::new()).expect("ifft2 gpu");
        let cpu_val = ifft2_builtin(Value::ComplexTensor(spectrum), Vec::new()).expect("ifft2 cpu");

        let gpu_ct = value_to_host_complex(gpu_val);
        let cpu_ct = value_to_host_complex(cpu_val);
        assert_eq!(gpu_ct.shape, cpu_ct.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (lhs, rhs) in gpu_ct
            .materialize_f64()
            .iter()
            .zip(cpu_ct.materialize_f64().iter())
        {
            assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
        }
        provider.free(&second).ok();
        runmat_accelerate_api::clear_residency(&second);
    }

    fn ifft2_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifft2_builtin(value, rest))
    }
}
