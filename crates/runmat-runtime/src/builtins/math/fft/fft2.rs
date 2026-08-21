//! MATLAB-compatible `fft2` builtin with GPU-aware semantics for RunMat.

use super::common::{
    download_provider_complex_tensor, ensure_wide_integer_data_exact, gather_gpu_complex_tensor,
    is_wide_integer_value, parse_2d_lengths_from_data, parse_2d_lengths_from_tensor, parse_length,
    restore_complex_gpu_result, transform_axes_complex_tensor, value_to_complex_tensor,
    TransformDirection,
};
use super::fft::fft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::fft::type_resolvers::fft2_type;
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
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fft2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fft2",
    op_kind: GpuOpKind::Custom("fft2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs two sequential `fft_dim` passes (dimensions 0 and 1); falls back to host execution when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fft2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fft2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "fft2 terminates fusion plans; fused kernels are not generated for multi-dimensional FFTs.",
};

const BUILTIN_NAME: &str = "fft2";

const FFT2_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2WideIntegerDataExtension"),
};
const FFT2_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 with int64 or uint64 transform-size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2WideIntegerControlExtension"),
};
const FFT2_SIZE_FORM_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-size-form",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2(X, SIZE) scalar/vector shorthand is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2SizeFormExtension"),
};
const FFT2_EMPTY_ZERO_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fft2-empty-zero-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fft2 empty or zero transform sizes are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Fft2EmptyZeroSizeExtension"),
};
pub const FFT2_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FFT2_WIDE_DATA_EXTENSION,
    FFT2_WIDE_CONTROL_EXTENSION,
    FFT2_SIZE_FORM_EXTENSION,
    FFT2_EMPTY_ZERO_SIZE_EXTENSION,
];

const FFT2_DOCUMENTED_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented integer arrays enter the double two-dimensional FFT domain.",
    }];
const FFT2_DOCUMENTED_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "M and N",
        classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "M and N are exact positive scalar structural controls; logical scalars are also documented.",
    }];
const FFT2_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Host wide data has its own gate and must cross the double boundary exactly.",
}];
const FFT2_WIDE_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "M, N, or SIZE",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "Wide structural controls have their own gate and are parsed exactly from authoritative storage.",
}];
const FFT2_SIZE_EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "SIZE",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "The scalar or two-element SIZE shorthand is independently mode-gated and parsed from authoritative storage.",
    }];
pub const FFT2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 5] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = fft2(integer_X)", inputs: &FFT2_DOCUMENTED_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer data produces double output; GPU fallback restores residency to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fft2(X, integer_M, integer_N)", inputs: &FFT2_DOCUMENTED_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Documented positive controls are parsed exactly before allocation or provider execution." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fft2(int64_or_uint64_X, ...)", inputs: &FFT2_WIDE_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only wide host data may not silently round; resident wide data is rejected without an exact provider transform." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fft2(X, int64_or_uint64_M_or_N_or_SIZE)", inputs: &FFT2_WIDE_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only wide controls are independently gated and decoded exactly before execution." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fft2(X, integer_SIZE)", inputs: &FFT2_SIZE_EXTENSION_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only scalar/two-vector shorthand; empty and zero sizes have a further independent extension gate." },
];

const FFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D complex Fourier spectrum output.",
}];

const FFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const FFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "RunMat-only scalar N or two-element [M N] size shorthand.",
    },
];

const FFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive output column count for transform.",
    },
];

const FFT2_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X)",
        inputs: &FFT2_INPUTS_CORE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, SIZE)",
        inputs: &FFT2_INPUTS_SIZE,
        outputs: &FFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fft2(X, M, N)",
        inputs: &FFT2_INPUTS_M_N,
        outputs: &FFT2_OUTPUT,
    },
];

const FFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.ARG_COUNT",
    identifier: Some("RunMat:fft2:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "fft2: invalid argument count",
};

const FFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_LENGTH",
    identifier: Some("RunMat:fft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "fft2: invalid transform length argument",
};

const FFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:fft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "fft2: invalid size vector argument",
};

const FFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INVALID_INPUT",
    identifier: Some("RunMat:fft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fft2: invalid input",
};

const FFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFT2.INTERNAL",
    identifier: Some("RunMat:fft2:Internal"),
    when: "FFT2 execution or tensor shaping fails.",
    message: "fft2: internal error",
};

const FFT2_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FFT2_ERROR_ARG_COUNT,
    FFT2_ERROR_INVALID_LENGTH,
    FFT2_ERROR_INVALID_SIZE_VECTOR,
    FFT2_ERROR_INVALID_INPUT,
    FFT2_ERROR_INTERNAL,
];

pub const FFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFT2_ERRORS,
};

fn fft2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fft2_error_with_message(error.message, error)
}

fn fft2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fft2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fft2_error_with_source(
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

fn fft2_error_with_message(
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
    name = "fft2",
    category = "math/fft",
    summary = "Compute two-dimensional discrete Fourier transforms.",
    keywords = "fft2,2d fft,two-dimensional fourier transform,gpu",
    type_resolver(fft2_type),
    descriptor(crate::builtins::math::fft::fft2::FFT2_DESCRIPTOR),
    extensions(crate::builtins::math::fft::fft2::FFT2_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::fft2::FFT2_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::fft2"
)]
async fn fft2_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "fft2")?;
    if is_wide_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFT2_WIDE_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
        ensure_wide_integer_data_exact(&value, BUILTIN_NAME)?;
    }
    if rest.iter().any(is_wide_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFT2_WIDE_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.len() == 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFT2_SIZE_FORM_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let lengths = parse_fft2_arguments(&rest)?;
    if !rest.is_empty()
        && (lengths.0.is_none()
            || lengths.1.is_none()
            || lengths.0 == Some(0)
            || lengths.1 == Some(0))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFT2_EMPTY_ZERO_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    match value {
        Value::GpuTensor(handle) => fft2_gpu(handle, lengths).await,
        other => fft2_host(other, lengths),
    }
}

fn fft2_host(value: Value, lengths: (Option<usize>, Option<usize>)) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        fft2_error_with_source(&FFT2_ERROR_INVALID_INPUT, "input conversion failed", source)
    })?;
    let transformed = fft2_complex_tensor(tensor, lengths)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<Value> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return fft2_gpu_fallback(handle, lengths).await;
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(first) = provider.fft_dim(&handle, lengths.0, 0).await {
            match provider.fft_dim(&first, lengths.1, 1).await {
                Ok(second) => {
                    if first.buffer_id != handle.buffer_id && first.buffer_id != second.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    return Ok(Value::GpuTensor(second));
                }
                Err(_) => {
                    let partial =
                        download_provider_complex_tensor(provider, &first, BUILTIN_NAME, false)
                            .await
                            .map_err(|source| {
                                fft2_error_with_source(
                                    &FFT2_ERROR_INVALID_INPUT,
                                    "provider download failed",
                                    source,
                                )
                            });
                    if first.buffer_id != handle.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    let completed = fft_complex_tensor(partial?, lengths.1, Some(2))?;
                    return restore_complex_gpu_result(&handle, &completed, BUILTIN_NAME);
                }
            }
        }
    }

    fft2_gpu_fallback(handle, lengths).await
}

async fn fft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<Value> {
    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            fft2_error_with_source(&FFT2_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = fft2_complex_tensor(complex, lengths)?;
    restore_complex_gpu_result(&handle, &transformed, BUILTIN_NAME)
}

fn fft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<ComplexTensor> {
    let (len_rows, len_cols) = lengths;
    transform_axes_complex_tensor(
        tensor,
        &[len_rows, len_cols],
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| fft2_error_with_source(&FFT2_ERROR_INTERNAL, "transform failed", source))
}

fn parse_fft2_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match args.len() {
        0 => Ok((None, None)),
        1 => parse_fft2_single(&args[0]),
        2 => {
            let rows = parse_length(&args[0], BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(
                    &FFT2_ERROR_INVALID_LENGTH,
                    "row-length parse failed",
                    source,
                )
            })?;
            let cols = parse_length(&args[1], BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(
                    &FFT2_ERROR_INVALID_LENGTH,
                    "column-length parse failed",
                    source,
                )
            })?;
            Ok((rows, cols))
        }
        _ => Err(fft2_error(&FFT2_ERROR_ARG_COUNT)),
    }
}

fn parse_fft2_single(value: &Value) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match value {
        Value::Tensor(tensor) => {
            parse_2d_lengths_from_tensor(tensor, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_detail(
                    &FFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical).map_err(|source| {
                fft2_error_with_detail(
                    &FFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("logical size-vector conversion failed: {source}"),
                )
            })?;
            parse_2d_lengths_from_data(&tensor::tensor_into_values_f64(tensor), BUILTIN_NAME)
                .map_err(|source| {
                    fft2_error_with_detail(
                        &FFT2_ERROR_INVALID_SIZE_VECTOR,
                        format!("size vector parse failed: {source}"),
                    )
                })
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let len = parse_length(value, BUILTIN_NAME).map_err(|source| {
                fft2_error_with_source(&FFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::Complex(_, _) => Err(fft2_error(&FFT2_ERROR_INVALID_LENGTH)),
        Value::ComplexTensor(_) => Err(fft2_error(&FFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::GpuTensor(_) => Err(fft2_error(&FFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
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
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => Err(fft2_error(&FFT2_ERROR_INVALID_LENGTH)),
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
    use runmat_builtins::{builtin_function_by_name, ResolveContext, Type};
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    fn value_to_host_complex(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu fft2 output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[test]
    fn fft2_type_pads_rank() {
        let out = fft2_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn fft2_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fft2 builtin");
        let descriptor = builtin.descriptor.expect("fft2 descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fft2(X)"));
        assert!(labels.contains(&"Y = fft2(X, SIZE)"));
        assert!(labels.contains(&"Y = fft2(X, M, N)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFT2.INVALID_LENGTH"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_matches_sequential_fft() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("fft2");
        let sequential = {
            let complex = value_to_complex_tensor(Value::Tensor(tensor), "fft2").unwrap();
            let first = fft_complex_tensor(complex, None, Some(1)).unwrap();
            fft_complex_tensor(first, None, Some(2)).unwrap()
        };
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, sequential.shape);
                for (lhs, rhs) in out
                    .materialize_f64()
                    .iter()
                    .zip(sequential.materialize_f64().iter())
                {
                    assert!(approx_eq(*lhs, *rhs, 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_scalar_length() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new((0..9).map(|v| v as f64).collect(), vec![3, 3]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 4]);
                assert_eq!(out.materialize_f64().len(), 16);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_size_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(size)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_size_vector_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let size = Tensor::new_integer(IntegerStorage::U16(vec![4, 2]), vec![1, 2]).unwrap();

        let result = fft2_builtin(Value::Tensor(tensor), vec![Value::Tensor(size)]).expect("fft2");

        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_empty_length_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(empty)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, tensor.shape),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_zero_length_returns_empty() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(3))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.materialize_f64().is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft2_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft2 gpu");
            let cpu = fft2_builtin(Value::Tensor(tensor), Vec::new()).expect("fft2 cpu");
            let g = value_to_host_complex(gpu);
            let c = value_to_host_complex(cpu);
            assert_eq!(g.shape, c.shape);
            let tol = 1e-10;
            for (lhs, rhs) in g.materialize_f64().iter().zip(c.materialize_f64().iter()) {
                assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
            }
            let original = test_support::gather(Value::GpuTensor(handle))
                .expect("caller-owned FFT input remains valid");
            assert_eq!(
                original.materialize_f64(),
                (0..8).map(|v| v as f64).collect::<Vec<_>>()
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_rejects_size_vector_with_more_than_two_entries() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0, 1.0], vec![1, 3]).unwrap();
        let err = error_message(
            fft2_builtin(Value::Tensor(tensor), vec![Value::Tensor(size)]).unwrap_err(),
        );
        assert!(err.contains("fft2"));
        assert!(err.contains("two elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_logical_scalar_size_extension() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(Value::Tensor(tensor), vec![Value::Bool(true)])
            .expect("logical scalar SIZE extension");
        assert!(matches!(result, Value::Complex(_, _)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_accepts_mixed_empty_and_length_arguments() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty), Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![2, 4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_integer_contract_separates_documented_controls_from_extensions() {
        let builtin = builtin_function_by_name("fft2").expect("fft2 registration");
        assert_eq!(builtin.integer_capabilities.len(), 5);
        assert_eq!(builtin.extensions.len(), 4);

        let logical = fft2_builtin(
            Value::Int(runmat_value::IntValue::U8(3)),
            vec![Value::Bool(true), Value::Bool(true)],
        )
        .expect("logical M and N are documented");
        assert!(matches!(logical, Value::Complex(_, _)));

        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let size_form = fft2_builtin(
            Value::Num(1.0),
            vec![Value::Int(runmat_value::IntValue::U16(2))],
        )
        .expect_err("single SIZE form must gate");
        assert_eq!(
            size_form.identifier(),
            Some("RunMat:compatibility:Fft2SizeFormExtension")
        );
        let zero = fft2_builtin(Value::Num(1.0), vec![Value::Num(0.0), Value::Num(2.0)])
            .expect_err("zero M must gate");
        assert_eq!(
            zero.identifier(),
            Some("RunMat:compatibility:Fft2EmptyZeroSizeExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fft2_rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fft2_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap_err();
        assert_eq!(error_identifier(&err), FFT2_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(FFT2_ERROR_ARG_COUNT.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fft2_wgpu_matches_cpu() {
        let provider = match std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
        }) {
            Ok(Ok(Some(provider))) => provider,
            _ => return,
        };

        let tensor = Tensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).expect("tensor");
        let tensor_cpu = tensor.clone();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value =
            fft2_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft2 gpu");
        let cpu_value = fft2_builtin(Value::Tensor(tensor_cpu), Vec::new()).expect("fft2 cpu");
        let gpu_ct = value_to_host_complex(gpu_value);
        let cpu_ct = value_to_host_complex(cpu_value);
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
        provider.free(&handle).ok();
        runmat_accelerate_api::clear_residency(&handle);
    }

    fn fft2_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fft2_builtin(value, rest))
    }
}
