//! MATLAB-compatible `hilbert` builtin for analytic signal construction.

use std::mem::size_of;

use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderHilbertRequest};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

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
use crate::builtins::math::fft::type_resolvers::fft_type;
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

const HILBERT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Analytic signal with real part equal to the input signal.",
}];

const HILBERT_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real-valued signal vector, matrix, or N-D array.",
}];

const HILBERT_INPUTS_WITH_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real-valued signal vector, matrix, or N-D array.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "FFT length along the first non-singleton dimension.",
    },
];

const HILBERT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x)",
        inputs: &HILBERT_INPUTS_CORE,
        outputs: &HILBERT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x, N)",
        inputs: &HILBERT_INPUTS_WITH_N,
        outputs: &HILBERT_OUTPUT,
    },
];

const HILBERT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.ARG_COUNT",
    identifier: Some("RunMat:hilbert:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "hilbert: expected hilbert(X) or hilbert(X, N)",
};

const HILBERT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_LENGTH",
    identifier: Some("RunMat:hilbert:InvalidLength"),
    when: "Length argument N is non-scalar, negative, non-finite, or fractional.",
    message: "hilbert: invalid length argument",
};

const HILBERT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_INPUT",
    identifier: Some("RunMat:hilbert:InvalidInput"),
    when: "Input cannot be converted to a real numeric/logical signal.",
    message: "hilbert: expected real numeric input",
};

const HILBERT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INTERNAL",
    identifier: Some("RunMat:hilbert:Internal"),
    when: "FFT execution or tensor shaping fails internally.",
    message: "hilbert: internal error",
};

const HILBERT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    HILBERT_ERROR_ARG_COUNT,
    HILBERT_ERROR_INVALID_LENGTH,
    HILBERT_ERROR_INVALID_INPUT,
    HILBERT_ERROR_INTERNAL,
];

const HILBERT_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with integer signal data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertIntegerDataExtension"),
};
const HILBERT_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with logical signal data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertLogicalDataExtension"),
};
const HILBERT_TYPED_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-typed-integer-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with a typed-integer FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertTypedIntegerLengthExtension"),
};
const HILBERT_LOGICAL_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-logical-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with a logical FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertLogicalLengthExtension"),
};
const HILBERT_ZERO_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-zero-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with a zero FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertZeroLengthExtension"),
};
const HILBERT_EMPTY_LENGTH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-empty-length",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with an empty FFT length is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertEmptyLengthExtension"),
};
const HILBERT_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "hilbert-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "hilbert with N-D input beyond a matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HilbertNdInputExtension"),
};
const HILBERT_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    HILBERT_INTEGER_DATA_EXTENSION,
    HILBERT_LOGICAL_DATA_EXTENSION,
    HILBERT_TYPED_LENGTH_EXTENSION,
    HILBERT_LOGICAL_LENGTH_EXTENSION,
    HILBERT_ZERO_LENGTH_EXTENSION,
    HILBERT_EMPTY_LENGTH_EXTENSION,
    HILBERT_ND_INPUT_EXTENSION,
];

const HILBERT_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "xr",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight real integer classes are admitted only in RunMat extension mode and must cross the binary64 transform boundary exactly.",
    }];
const HILBERT_INTEGER_LENGTH_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Typed-integer n is a RunMat-only structural control parsed exactly before allocation.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = hilbert(integer_xr[, n])",
        inputs: &HILBERT_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat validates exact binary64 conversion before the FFT-domain analytic-signal computation; resident input gathers through and restores to its owning provider when binary64 is supported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = hilbert(xr, integer_n)",
        inputs: &HILBERT_INTEGER_LENGTH_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed n preserves exact integer control semantics; ordinary output class follows the floating signal input.",
    },
];

pub const HILBERT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HILBERT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HILBERT_ERRORS,
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
    category = "math/signal",
    summary = "Construct the analytic signal with the Hilbert transform.",
    keywords = "hilbert,analytic signal,signal processing,fft,complex",
    type_resolver(fft_type),
    descriptor(crate::builtins::math::signal::hilbert::HILBERT_DESCRIPTOR),
    extensions(HILBERT_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::hilbert::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::hilbert"
)]
async fn hilbert_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_hilbert_extensions(&value, &rest)?;
    let length = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => hilbert_gpu_tensor(handle, length).await,
        Value::Complex(re, _) => hilbert_tensor(
            Tensor::new(vec![re], vec![1, 1]).map_err(|source| {
                hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, source.to_string())
            })?,
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
    }
}

async fn hilbert_gpu_tensor(
    handle: GpuTensorHandle,
    length: Option<usize>,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        hilbert_terminal_error(
            &HILBERT_ERROR_INTERNAL,
            "GPU provider unavailable for input",
        )
    })?;
    let complex_input =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;
    let integer_input = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    let logical_input = runmat_accelerate_api::handle_is_logical(&handle);

    let mut shape = handle.shape.clone();
    if crate::builtins::common::shape::is_scalar_shape(&shape) {
        shape = crate::builtins::common::shape::normalize_scalar_shape(&shape);
    }
    let dim_one_based = default_dimension(&shape);
    let dim_index = dim_one_based - 1;
    validate_transform_allocation(&shape, dim_index, length)?;

    let current_len = shape.get(dim_index).copied().unwrap_or(1);
    let target_len = length.unwrap_or(current_len);
    if target_len != 0 && !complex_input && !integer_input && !logical_input {
        let input_metadata = hilbert_input_metadata(&handle);
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        match provider
            .signal_hilbert(&ProviderHilbertRequest {
                input: &handle,
                length,
                dim: dim_index,
            })
            .await
        {
            Ok(out)
                if valid_hilbert_gpu_output(
                    &out,
                    &handle,
                    provider,
                    &logical_shape_for_hilbert(&shape, dim_index, target_len),
                    runmat_accelerate_api::handle_precision(&handle)
                        .unwrap_or_else(|| provider.precision()),
                ) =>
            {
                return Ok(gpu_helpers::complex_gpu_value(out));
            }
            Ok(out) => {
                if hilbert_gpu_handles_alias(&out, &handle) {
                    restore_hilbert_input_metadata(&handle, input_metadata);
                }
                free_rejected_hilbert_output(&out, &handle);
                return Err(hilbert_terminal_error(
                    &HILBERT_ERROR_INTERNAL,
                    "provider signal_hilbert returned malformed output",
                ));
            }
            Err(err) => {
                if !hilbert_provider_operation_unsupported(&err, "signal_hilbert") {
                    return Err(hilbert_terminal_error(
                        &HILBERT_ERROR_INTERNAL,
                        format!("provider signal_hilbert failed: {err}"),
                    ));
                }
            }
        }
    }

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
        Value::ComplexTensor(tensor) => {
            hilbert_tensor(real_part_tensor_with_precision(tensor, &handle)?, length)?
        }
        Value::Complex(re, _) => hilbert_tensor(
            Tensor::new(vec![re], shape.clone()).map_err(|source| {
                hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, source.to_string())
            })?,
            length,
        )?,
        other => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, other).map_err(|detail| {
                hilbert_error_with_detail(&HILBERT_ERROR_INVALID_INPUT, detail)
            })?;
            hilbert_tensor(tensor, length)?
        }
    };
    restore_hilbert_gpu_output(provider, &handle, host)
}

fn hilbert_provider_operation_unsupported(error: &anyhow::Error, operation: &str) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string() == format!("{operation} not supported by provider"))
}

fn ensure_hilbert_extensions(value: &Value, args: &[Value]) -> BuiltinResult<()> {
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
        Value::Tensor(tensor) => tensor.shape.len(),
        Value::ComplexTensor(tensor) => tensor.shape.len(),
        Value::LogicalArray(array) => array.shape.len(),
        Value::GpuTensor(handle) => handle.shape.len(),
        _ => 2,
    };
    if rank > 2 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HILBERT_ND_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if let [length] = args {
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
    let shape = tensor.shape.clone();
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
        .map_err(|source| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, source.to_string()))
}

fn real_part_tensor_with_precision(
    tensor: ComplexTensor,
    input: &GpuTensorHandle,
) -> BuiltinResult<Tensor> {
    if runmat_accelerate_api::handle_precision(input)
        != Some(runmat_accelerate_api::ProviderPrecision::F32)
    {
        return real_part_tensor(tensor);
    }
    let shape = tensor.shape.clone();
    let values = tensor
        .materialize_f64()
        .iter()
        .map(|(real, _)| *real as f32)
        .collect();
    Tensor::from_f32(values, shape)
        .map_err(|source| hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, source.to_string()))
}

fn logical_shape_for_hilbert(shape: &[usize], dim: usize, length: usize) -> Vec<usize> {
    let mut output = shape.to_vec();
    while output.len() <= dim {
        output.push(1);
    }
    output[dim] = length;
    output
}

fn restore_hilbert_gpu_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    value: Value,
) -> BuiltinResult<Value> {
    let tensor = match value {
        Value::ComplexTensor(tensor) => tensor,
        Value::Complex(real, imag) => {
            ComplexTensor::new(vec![(real, imag)], vec![1, 1]).map_err(|source| {
                hilbert_error_with_detail(&HILBERT_ERROR_INTERNAL, source.to_string())
            })?
        }
        other => {
            return Err(hilbert_terminal_error(
                &HILBERT_ERROR_INTERNAL,
                format!("unexpected host fallback result {other:?}"),
            ));
        }
    };
    let expected_precision = match tensor.numeric_dtype() {
        runmat_builtins::NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        _ => runmat_accelerate_api::ProviderPrecision::F64,
    };
    if provider.precision() != expected_precision {
        return Ok(Value::ComplexTensor(tensor));
    }
    let expected_shape = tensor.shape.clone();
    let input_metadata = hilbert_input_metadata(input);
    let output = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|source| {
        hilbert_terminal_error(
            &HILBERT_ERROR_INTERNAL,
            format!("failed to restore fallback result to input provider: {source}"),
        )
    })?;
    if !valid_hilbert_gpu_output(
        &output,
        input,
        provider,
        &expected_shape,
        expected_precision,
    ) {
        if hilbert_gpu_handles_alias(&output, input) {
            restore_hilbert_input_metadata(input, input_metadata);
        }
        free_rejected_hilbert_output(&output, input);
        return Err(hilbert_terminal_error(
            &HILBERT_ERROR_INTERNAL,
            "provider upload returned malformed fallback output",
        ));
    }
    Ok(gpu_helpers::complex_gpu_value(output))
}

fn valid_hilbert_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    expected_precision: runmat_accelerate_api::ProviderPrecision,
) -> bool {
    output.shape == expected_shape
        && output.device_id == input.device_id
        && !hilbert_gpu_handles_alias(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::ComplexInterleaved
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

type HilbertGpuMetadata = (
    GpuTensorStorage,
    Option<runmat_accelerate_api::ProviderPrecision>,
    Option<runmat_accelerate_api::IntegerElementType>,
    bool,
);

fn hilbert_input_metadata(handle: &GpuTensorHandle) -> HilbertGpuMetadata {
    (
        runmat_accelerate_api::handle_storage(handle),
        runmat_accelerate_api::handle_precision(handle),
        runmat_accelerate_api::handle_integer_type(handle),
        runmat_accelerate_api::handle_is_logical(handle),
    )
}

fn restore_hilbert_input_metadata(handle: &GpuTensorHandle, metadata: HilbertGpuMetadata) {
    if handle.descriptor.storage.is_none() {
        runmat_accelerate_api::set_handle_storage(handle, metadata.0);
    } else {
        runmat_accelerate_api::clear_handle_storage(handle);
    }
    if handle.descriptor.element_type.is_none() {
        if let Some(precision) = metadata.1 {
            runmat_accelerate_api::set_handle_precision(handle, precision);
        } else {
            runmat_accelerate_api::clear_handle_precision(handle);
        }
        if let Some(integer) = metadata.2 {
            runmat_accelerate_api::set_handle_integer_type(handle, integer);
        } else {
            runmat_accelerate_api::clear_handle_integer_type(handle);
        }
    } else {
        runmat_accelerate_api::clear_handle_precision(handle);
        runmat_accelerate_api::clear_handle_integer_type(handle);
    }
    runmat_accelerate_api::set_handle_logical(handle, metadata.3);
}

fn hilbert_gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_hilbert_output(output: &GpuTensorHandle, input: &GpuTensorHandle) {
    if hilbert_gpu_handles_alias(output, input) {
        return;
    }
    if let Some(owner) = runmat_accelerate_api::provider_for_handle(output) {
        if owner.free(output).is_ok() {
            runmat_accelerate_api::clear_residency(output);
        }
    }
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<Option<usize>> {
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
    let mut shape = tensor.shape.clone();
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
    let mut shape = spectrum.shape.clone();
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

    ComplexTensor::from_f64_values_with_dtype(data, spectrum.shape, dtype)
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
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor as HostComplexTensor, IntValue, LogicalArray, Type};

    const TOL: f64 = 1.0e-12;

    fn hilbert_call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(hilbert_builtin(value, rest))
    }

    fn as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            Value::Complex(re, im) => HostComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
            Value::GpuTensor(handle) => block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(
                    &handle,
                    BUILTIN_NAME,
                ),
            )
            .expect("gather complex gpu output"),
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    fn assert_complex_close(actual: (f64, f64), expected: (f64, f64)) {
        assert!(
            (actual.0 - expected.0).abs() <= TOL,
            "real mismatch: actual={} expected={}",
            actual.0,
            expected.0
        );
        assert!(
            (actual.1 - expected.1).abs() <= TOL,
            "imag mismatch: actual={} expected={}",
            actual.1,
            expected.1
        );
    }

    #[test]
    fn hilbert_type_preserves_numeric_shape() {
        let out = fft_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(4)]),
            }],
            &runmat_builtins::ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn hilbert_row_cosine_returns_quadrature_signal() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 4]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_column_cosine_operates_down_columns() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![4, 1]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![4, 1]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_matrix_operates_along_first_nonsingleton_dimension() {
        let input =
            Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0], vec![4, 2]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::Tensor(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![4, 2]);
        let expected = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
        ];
        for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_length_argument_pads_or_truncates_transform_axis() {
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out =
            as_complex_tensor(hilbert_call(Value::Tensor(input), vec![Value::Num(6.0)]).unwrap());
        assert_eq!(out.shape, vec![1, 6]);
        assert_eq!(out.materialize_f64().len(), 6);
        assert_complex_close(out.materialize_f64()[0], (1.0, 0.0));
    }

    #[test]
    fn hilbert_zero_length_returns_empty_along_transform_axis() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(
            hilbert_call(Value::Tensor(input), vec![Value::Int(IntValue::I32(0))]).unwrap(),
        );
        assert_eq!(out.shape, vec![1, 0]);
        assert!(out.materialize_f64().is_empty());
    }

    #[test]
    fn hilbert_accepts_logical_input_as_real_signal() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = LogicalArray::new(vec![1, 0, 1, 0], vec![1, 4]).unwrap();
        let out = as_complex_tensor(hilbert_call(Value::LogicalArray(input), Vec::new()).unwrap());
        assert_eq!(out.shape, vec![1, 4]);
    }

    #[test]
    fn hilbert_ignores_complex_input_imaginary_part() {
        let input = HostComplexTensor::new(
            vec![(1.0, 9.0), (0.0, -4.0), (-1.0, 2.0), (0.0, 7.0)],
            vec![1, 4],
        )
        .unwrap();
        let output =
            as_complex_tensor(hilbert_call(Value::ComplexTensor(input), Vec::new()).unwrap());
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in output.materialize_f64().iter().copied().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn hilbert_runmat_extensions_follow_compatibility_mode() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let cases = [
            (
                Value::Int(IntValue::I32(2)),
                "RunMat:compatibility:HilbertTypedIntegerLengthExtension",
            ),
            (
                Value::Bool(true),
                "RunMat:compatibility:HilbertLogicalLengthExtension",
            ),
            (
                Value::Num(0.0),
                "RunMat:compatibility:HilbertZeroLengthExtension",
            ),
        ];
        for (length, identifier) in cases {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = hilbert_call(Value::Tensor(input.clone()), vec![length])
                .expect_err("strict mode rejects extension");
            assert_eq!(err.identifier(), Some(identifier));
        }
    }

    #[test]
    fn hilbert_rejects_fractional_length() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let err = hilbert_call(Value::Tensor(input), vec![Value::Num(1.5)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:hilbert:InvalidLength"));
    }

    #[test]
    fn hilbert_rejects_huge_length_before_allocation() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let err = hilbert_call(Value::Tensor(input), vec![Value::Num(f64::MAX)]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:hilbert:InvalidLength"));
    }

    #[test]
    fn hilbert_gpu_input_returns_owner_resident_analytic_signal() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &input.materialize_f64(),
                shape: &input.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let metadata = hilbert_input_metadata(&handle);
            let out = as_complex_tensor(
                hilbert_call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap(),
            );
            assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
            assert_eq!(hilbert_input_metadata(&handle), metadata);
            assert_eq!(out.shape, vec![1, 4]);
            let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
            for (actual, expected) in out.materialize_f64().iter().copied().zip(expected) {
                assert_complex_close(actual, expected);
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn hilbert_wgpu_input_returns_resident_complex_gpu_tensor() {
        use crate::builtins::common::test_support;
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::AccelProvider;

        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &input.materialize_f64(),
            shape: &input.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let out = hilbert_call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap();
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident GPU output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );

        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(
                &out_handle,
                BUILTIN_NAME,
            ),
        )
        .expect("gather complex output");
        assert_eq!(gathered.shape, vec![1, 4]);
        let expected = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        for (actual, expected) in gathered.materialize_f64().iter().copied().zip(expected) {
            assert!((actual.0 - expected.0).abs() <= 1.0e-5);
            assert!((actual.1 - expected.1).abs() <= 1.0e-5);
        }
        provider.free(&handle).ok();
        provider.free(&out_handle).ok();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn hilbert_wgpu_length_argument_matches_cpu_and_stays_resident() {
        use crate::builtins::common::test_support;
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::AccelProvider;

        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let input = Tensor::new(vec![1.0, 0.0, -1.0, 0.0], vec![1, 4]).unwrap();
        let expected = as_complex_tensor(
            hilbert_call(Value::Tensor(input.clone()), vec![Value::Num(6.0)]).unwrap(),
        );
        let view = runmat_accelerate_api::HostTensorView {
            data: &input.materialize_f64(),
            shape: &input.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let out = hilbert_call(Value::GpuTensor(handle.clone()), vec![Value::Num(6.0)]).unwrap();
        let Value::GpuTensor(out_handle) = out else {
            panic!("expected resident GPU output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );

        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(
                &out_handle,
                BUILTIN_NAME,
            ),
        )
        .expect("gather complex output");
        assert_eq!(gathered.shape, expected.shape);
        for (idx, (actual, expected)) in gathered
            .materialize_f64()
            .iter()
            .copied()
            .zip(expected.materialize_f64())
            .enumerate()
        {
            assert!(
                (actual.0 - expected.0).abs() <= 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() <= 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
        provider.free(&handle).ok();
        provider.free(&out_handle).ok();
    }
}
