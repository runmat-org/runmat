//! MATLAB-compatible `fftn` builtin with GPU-aware semantics for RunMat.

use super::common::{
    download_provider_complex_tensor, ensure_wide_integer_data_exact, gather_gpu_complex_tensor,
    is_wide_integer_value, parse_nd_sizes_value, restore_complex_gpu_result,
    transform_complex_tensor, transform_nd_complex_tensor, value_to_complex_tensor,
    TransformDirection,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::fft::type_resolvers::fftn_type;
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fftn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftn",
    op_kind: GpuOpKind::Custom("fftn"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs sequential `fft_dim` passes along each transformed axis; falls back to host execution when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fftn")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fftn terminates fusion plans; fused kernels are not generated for N-D FFTs.",
};

const BUILTIN_NAME: &str = "fftn";

const FFTN_WIDE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-wide-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn with host int64 or uint64 data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnWideIntegerDataExtension"),
};
const FFTN_WIDE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-wide-integer-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn with int64 or uint64 SIZE controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnWideIntegerControlExtension"),
};
const FFTN_SHORT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fftn-short-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fftn SIZE vectors shorter than ndims(X) are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FftnShortSizeExtension"),
};
pub const FFTN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    FFTN_WIDE_DATA_EXTENSION,
    FFTN_WIDE_CONTROL_EXTENSION,
    FFTN_SHORT_SIZE_EXTENSION,
];

const FFTN_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Documented integer arrays enter the double N-dimensional FFT domain.",
}];
const FFTN_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE", classes: &crate::builtins::common::integer_capability::INTEGER_CLASSES_THROUGH_32_BITS,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "SIZE is a positive integer vector with length at least ndims(X); logical vectors are also documented.",
}];
const FFTN_WIDE_DATA_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Host wide data has its own gate and must be exactly representable as double.",
}];
const FFTN_WIDE_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "SIZE",
    classes: &[
        runmat_builtins::BuiltinIntegerClass::Int64,
        runmat_builtins::BuiltinIntegerClass::Uint64,
    ],
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes:
        "Wide SIZE controls have their own gate and are decoded exactly from authoritative storage.",
}];
pub const FFTN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftn(integer_X)", inputs: &FFTN_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Documented integer data produces double output and resident fallback returns to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftn(X, integer_SIZE)", inputs: &FFTN_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "SIZE is decoded exactly and validated for vector shape, positivity, and minimum length before execution." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftn(int64_or_uint64_X)", inputs: &FFTN_WIDE_DATA_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat-only wide host data may not silently round; resident wide data is rejected without an exact provider transform." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = fftn(X, int64_or_uint64_SIZE)", inputs: &FFTN_WIDE_SIZE_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat-only wide SIZE controls are independently gated; short positive vectors have a separate extension gate." },
];

const FFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D complex Fourier spectrum output.",
}];

const FFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const FFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
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
        description: "Positive transform-size vector with length at least ndims(X).",
    },
];

const FFTN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X)",
        inputs: &FFTN_INPUTS_CORE,
        outputs: &FFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X, SIZE)",
        inputs: &FFTN_INPUTS_SIZE,
        outputs: &FFTN_OUTPUT,
    },
];

const FFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.ARG_COUNT",
    identifier: Some("RunMat:fftn:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftn: invalid argument count",
};

const FFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_SIZE",
    identifier: Some("RunMat:fftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "fftn: invalid SIZE argument",
};

const FFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_INPUT",
    identifier: Some("RunMat:fftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fftn: invalid input",
};

const FFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INTERNAL",
    identifier: Some("RunMat:fftn:Internal"),
    when: "FFTN execution or tensor shaping fails.",
    message: "fftn: internal error",
};

const FFTN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FFTN_ERROR_ARG_COUNT,
    FFTN_ERROR_INVALID_SIZE,
    FFTN_ERROR_INVALID_INPUT,
    FFTN_ERROR_INTERNAL,
];

pub const FFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTN_ERRORS,
};

fn fftn_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fftn_error_with_message(error.message, error)
}

fn fftn_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fftn_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fftn_error_with_source(
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

fn fftn_error_with_message(
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
    name = "fftn",
    category = "math/fft",
    summary = "Compute the N-dimensional discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fftn,nd fft,n-dimensional fourier transform,gpu",
    type_resolver(fftn_type),
    descriptor(crate::builtins::math::fft::fftn::FFTN_DESCRIPTOR),
    extensions(crate::builtins::math::fft::fftn::FFTN_EXTENSIONS),
    integer_capabilities(crate::builtins::math::fft::fftn::FFTN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::fft::fftn"
)]
async fn fftn_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "fftn")?;
    if is_wide_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFTN_WIDE_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
        ensure_wide_integer_data_exact(&value, BUILTIN_NAME)?;
    }
    if rest.iter().any(is_wide_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FFTN_WIDE_CONTROL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let sizes = parse_fftn_sizes(&rest, fftn_input_rank(&value))?;
    match value {
        Value::GpuTensor(handle) => fftn_gpu(handle, sizes).await,
        other => fftn_host(other, sizes),
    }
}

fn fftn_host(value: Value, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        fftn_error_with_source(&FFTN_ERROR_INVALID_INPUT, "input conversion failed", source)
    })?;
    let transformed = fftn_complex_tensor(tensor, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fftn_gpu(handle: GpuTensorHandle, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let mut current = handle.clone();
        let mut prior_handles = Vec::new();
        let mut logical_shape = current.shape.clone();
        if logical_shape.is_empty() {
            logical_shape.push(1);
        }
        let axis_count = sizes
            .as_ref()
            .map(|v| v.len())
            .unwrap_or_else(|| logical_shape.len());

        for axis in 0..axis_count {
            let len = sizes.as_ref().and_then(|v| v.get(axis).copied());
            match provider.fft_dim(&current, len, axis).await {
                Ok(next) => {
                    if current.buffer_id != next.buffer_id {
                        prior_handles.push(current);
                    }
                    current = next;
                }
                Err(_) => {
                    let downloaded =
                        download_provider_complex_tensor(provider, &current, BUILTIN_NAME, false)
                            .await
                            .map_err(|source| {
                                fftn_error_with_source(
                                    &FFTN_ERROR_INVALID_INPUT,
                                    "provider download failed",
                                    source,
                                )
                            });
                    let mut transformed = downloaded;
                    for remaining_axis in axis..axis_count {
                        let remaining_len = sizes
                            .as_ref()
                            .and_then(|values| values.get(remaining_axis).copied());
                        transformed = transformed.and_then(|partial| {
                            transform_complex_tensor(
                                partial,
                                remaining_len,
                                Some(remaining_axis + 1),
                                TransformDirection::Forward,
                                BUILTIN_NAME,
                            )
                        });
                    }
                    for prior in prior_handles {
                        if prior.buffer_id != handle.buffer_id {
                            provider.free(&prior).ok();
                            runmat_accelerate_api::clear_residency(&prior);
                        }
                    }
                    if current.buffer_id != handle.buffer_id {
                        provider.free(&current).ok();
                        runmat_accelerate_api::clear_residency(&current);
                    }
                    let partial = transformed?;
                    return restore_complex_gpu_result(&handle, &partial, BUILTIN_NAME);
                }
            }
        }
        for prior in prior_handles {
            if prior.buffer_id != handle.buffer_id && prior.buffer_id != current.buffer_id {
                provider.free(&prior).ok();
                runmat_accelerate_api::clear_residency(&prior);
            }
        }
        return Ok(Value::GpuTensor(current));
    }

    fftn_gpu_fallback(handle, sizes).await
}

async fn fftn_gpu_fallback(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let complex = download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false)
            .await
            .map_err(|source| {
                fftn_error_with_source(
                    &FFTN_ERROR_INVALID_INPUT,
                    "provider download failed",
                    source,
                )
            })?;
        let transformed = fftn_complex_tensor(complex, sizes)?;
        return restore_complex_gpu_result(&handle, &transformed, BUILTIN_NAME);
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            fftn_error_with_source(&FFTN_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = fftn_complex_tensor(complex, sizes)?;
    restore_complex_gpu_result(&handle, &transformed, BUILTIN_NAME)
}

fn fftn_complex_tensor(
    tensor: ComplexTensor,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<ComplexTensor> {
    transform_nd_complex_tensor(
        tensor,
        sizes.as_deref(),
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| fftn_error_with_source(&FFTN_ERROR_INTERNAL, "transform failed", source))
}

fn parse_fftn_sizes(args: &[Value], input_rank: usize) -> BuiltinResult<Option<Vec<usize>>> {
    match args.len() {
        0 => Ok(None),
        1 => {
            validate_size_vector_shape(&args[0])?;
            let sizes = parse_sizes_value(&args[0])?;
            if sizes.is_empty() || sizes.contains(&0) {
                return Err(fftn_error_with_detail(
                    &FFTN_ERROR_INVALID_SIZE,
                    "SIZE must contain positive integers",
                ));
            }
            if sizes.len() < input_rank {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FFTN_SHORT_SIZE_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(Some(sizes))
        }
        _ => Err(fftn_error(&FFTN_ERROR_ARG_COUNT)),
    }
}

fn fftn_input_rank(value: &Value) -> usize {
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
        Value::Tensor(tensor) if !vector_shape(&tensor.shape) => Err(fftn_error_with_detail(
            &FFTN_ERROR_INVALID_SIZE,
            "SIZE must be a row or column vector",
        )),
        Value::LogicalArray(array) if !vector_shape(&array.shape) => Err(fftn_error_with_detail(
            &FFTN_ERROR_INVALID_SIZE,
            "SIZE must be a row or column vector",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(fftn_error_with_detail(
            &FFTN_ERROR_INVALID_SIZE,
            "SIZE must be real-valued",
        )),
        _ => Ok(()),
    }
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    parse_nd_sizes_value(value, BUILTIN_NAME).map_err(|source| {
        fftn_error_with_detail(
            &FFTN_ERROR_INVALID_SIZE,
            format!("SIZE parse failed: {source}"),
        )
    })
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
    fn fftn_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftn builtin");
        let descriptor = builtin.descriptor.expect("fftn descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fftn(X)"));
        assert!(labels.contains(&"Y = fftn(X, SIZE)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFTN.INVALID_SIZE"));
    }

    #[test]
    fn fftn_matches_sequential_fft_on_3d() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input), BUILTIN_NAME).unwrap();
        let got = fftn_complex_tensor(complex.clone(), None).unwrap();

        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let expect = fft_complex_tensor(b, None, Some(3)).unwrap();

        assert_eq!(got.shape, expect.shape);
        for (lhs, rhs) in got
            .materialize_f64()
            .iter()
            .zip(expect.materialize_f64().iter())
        {
            assert!((lhs.0 - rhs.0).abs() < 1e-12);
            assert!((lhs.1 - rhs.1).abs() < 1e-12);
        }
    }

    #[test]
    fn fftn_rejects_invalid_argument_count() {
        let err = parse_fftn_sizes(&[Value::Num(2.0), Value::Num(3.0)], 2).unwrap_err();
        assert_eq!(error_identifier(&err), FFTN_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(FFTN_ERROR_ARG_COUNT.message));
    }

    #[test]
    fn fftn_rejects_invalid_size_argument() {
        let err = parse_fftn_sizes(&[Value::from("bad")], 2).unwrap_err();
        assert_eq!(error_identifier(&err), FFTN_ERROR_INVALID_SIZE.identifier);
        assert!(error_message(err).contains(FFTN_ERROR_INVALID_SIZE.message));
    }

    #[test]
    fn fftn_integer_contract_validates_documented_size_vectors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftn builtin");
        assert_eq!(builtin.integer_capabilities.len(), 4);
        assert_eq!(builtin.extensions.len(), 3);

        let logical_sizes =
            runmat_value::LogicalArray::new(vec![1, 1], vec![1, 2]).expect("logical size vector");
        block_on(fftn_builtin(
            Value::Int(runmat_value::IntValue::I16(4)),
            vec![Value::LogicalArray(logical_sizes)],
        ))
        .expect("documented logical SIZE vector");

        let matrix = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let error = block_on(fftn_builtin(Value::Num(1.0), vec![Value::Tensor(matrix)]))
            .expect_err("SIZE matrix must be rejected");
        assert_eq!(error.identifier(), FFTN_ERROR_INVALID_SIZE.identifier);

        let zero = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let error = block_on(fftn_builtin(Value::Num(1.0), vec![Value::Tensor(zero)]))
            .expect_err("zero SIZE must be rejected");
        assert_eq!(error.identifier(), FFTN_ERROR_INVALID_SIZE.identifier);
    }

    #[test]
    fn fftn_short_size_and_wide_data_extensions_gate_independently() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let short = block_on(fftn_builtin(
            Value::Num(1.0),
            vec![Value::Int(runmat_value::IntValue::U16(2))],
        ))
        .expect_err("short SIZE must gate");
        assert_eq!(
            short.identifier(),
            Some("RunMat:compatibility:FftnShortSizeExtension")
        );

        let wide =
            Tensor::new_integer(runmat_value::IntegerStorage::I64(vec![1]), vec![1, 1]).unwrap();
        let error = block_on(fftn_builtin(Value::Tensor(wide), Vec::new()))
            .expect_err("wide data must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FftnWideIntegerDataExtension")
        );
    }
}
