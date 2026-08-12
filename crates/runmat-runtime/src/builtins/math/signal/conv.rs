//! MATLAB-compatible `conv` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex;
use runmat_accelerate_api::{ProviderConv1dOptions, ProviderConvMode, ProviderConvOrientation};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::conv_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::conv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conv",
    op_kind: GpuOpKind::Custom("conv1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("conv1d")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses a native hook only for same-provider, same-device real floating inputs and accepts its result only when the handle has the required precision, real storage, input device, and provider ownership. Other supported inputs gather through their owners and restore only when class and complexity can be preserved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::conv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Convolution boundaries terminate fusion plans; intermediate expressions run on the host today.",
};

const BUILTIN_NAME: &str = "conv";

const CONV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Convolution result.",
}];

const CONV_SIG_AB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input vector/scalar.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input vector/scalar.",
    },
];

const CONV_SIG_SHAPE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input vector/scalar.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input vector/scalar.",
    },
    BuiltinParamDescriptor {
        name: "shape",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"full\""),
        description: "Output shape: \"full\", \"same\", or \"valid\".",
    },
];

const CONV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = conv(A, B)",
        inputs: &CONV_SIG_AB_INPUTS,
        outputs: &CONV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = conv(A, B, shape)",
        inputs: &CONV_SIG_SHAPE_INPUTS,
        outputs: &CONV_OUTPUT,
    },
];

const CONV_ERROR_SHAPE_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.SHAPE_INVALID",
    identifier: Some("RunMat:conv:ShapeInvalid"),
    when: "Third argument is missing or not one of full/same/valid.",
    message: "conv: third argument must be the string 'full', 'same', or 'valid'",
};

const CONV_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.ARG_COUNT",
    identifier: Some("RunMat:conv:ArgCount"),
    when: "More than three input arguments are provided.",
    message: "conv: expected at most three input arguments",
};

const CONV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.INVALID_INPUT",
    identifier: Some("RunMat:conv:InvalidInput"),
    when: "An input value is not numeric/logical scalar/vector/tensor compatible.",
    message: "conv: unsupported input type",
};

const CONV_ERROR_CONVERSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.CONVERSION",
    identifier: Some("RunMat:conv:Conversion"),
    when: "Input conversion from logical/gpu tensor into host numeric domain fails.",
    message: "conv: input conversion failed",
};

const CONV_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.GATHER_FAILED",
    identifier: Some("RunMat:conv:GatherFailed"),
    when: "GPU input cannot be gathered for host fallback normalization.",
    message: "conv: failed to gather GPU input",
};

const CONV_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.BUILD_OUTPUT",
    identifier: Some("RunMat:conv:BuildOutput"),
    when: "Output tensor allocation fails.",
    message: "conv: failed to build tensor",
};

const CONV_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONV.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:conv:BuildComplexOutput"),
    when: "Output complex tensor allocation fails.",
    message: "conv: failed to build complex tensor",
};

const CONV_ERRORS: [BuiltinErrorDescriptor; 7] = [
    CONV_ERROR_SHAPE_INVALID,
    CONV_ERROR_ARG_COUNT,
    CONV_ERROR_INVALID_INPUT,
    CONV_ERROR_CONVERSION,
    CONV_ERROR_GATHER_FAILED,
    CONV_ERROR_BUILD_OUTPUT,
    CONV_ERROR_BUILD_COMPLEX_OUTPUT,
];

pub const CONV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONV_ERRORS,
};

const CONV_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "u", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight real or complex integer classes are documented; values cross the explicit floating convolution boundary before multiplication and accumulation." },
    BuiltinIntegerInputCapability { name: "v", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Mixed integer classes are documented and independently converted to the selected floating output domain." },
];
pub const CONV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "w = conv(integer_u, integer_v, shape?)", inputs: &CONV_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "The output is single when either numeric input is single and double otherwise; integer and logical inputs never produce an integer result. Full orientation follows the R2026a both-columns rule, while same/valid follow u. Resident integer inputs gather exactly through their owning provider before floating convolution and the floating result is restored to that provider." }];

fn conv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    conv_error_with_message(error.message, error)
}

fn conv_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    conv_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn conv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn conv_error_with_source(
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

#[runtime_builtin(
    name = "conv",
    category = "math/signal",
    summary = "Compute one-dimensional linear convolution.",
    keywords = "conv,convolution,signal processing,gpu",
    accel = "custom",
    type_resolver(conv_type),
    integer_capabilities(CONV_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::signal::conv::CONV_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::conv"
)]
async fn conv_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mode = parse_mode(&rest)?;
    let single_output = value_uses_single(&a) || value_uses_single(&b);
    let complex_output = value_is_complex(&a) || value_is_complex(&b);
    let owner = first_gpu_handle(&a, &b);
    if let Some(device_value) = try_conv_gpu(&a, &b, mode, single_output)? {
        return Ok(device_value);
    }
    let lhs = normalize_input(a).await?;
    let rhs = normalize_input(b).await?;
    if matches!(lhs.hint, OrientationHint::General) || matches!(rhs.hint, OrientationHint::General)
    {
        return Err(conv_error_with_detail(
            &CONV_ERROR_INVALID_INPUT,
            "inputs must be vectors",
        ));
    }
    let orientation = output_orientation(&lhs, &rhs, mode);

    if lhs.len == 0 || rhs.len == 0 {
        let data = if matches!(mode, ConvMode::Valid) && rhs.len == 0 {
            vec![Complex::new(0.0, 0.0); lhs.len]
        } else {
            Vec::new()
        };
        let output = convert_output(data, orientation, single_output, complex_output)?;
        return restore_to_owner(output, owner.as_ref(), single_output);
    }

    let full = convolve(&lhs.data, &rhs.data, single_output);
    let shaped = apply_mode(full, mode, lhs.len, rhs.len);
    let output = convert_output(shaped, orientation, single_output, complex_output)?;
    restore_to_owner(output, owner.as_ref(), single_output)
}

fn first_gpu_handle(a: &Value, b: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    match (a, b) {
        (Value::GpuTensor(handle), _) | (_, Value::GpuTensor(handle)) => Some(handle.clone()),
        _ => None,
    }
}

fn restore_to_owner(
    value: Value,
    owner: Option<&runmat_accelerate_api::GpuTensorHandle>,
    single_output: bool,
) -> BuiltinResult<Value> {
    let Some(provider) = owner.and_then(runmat_accelerate_api::provider_for_handle) else {
        return Ok(value);
    };
    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    if provider.precision() != expected {
        return Ok(value);
    }
    match value {
        Value::Tensor(tensor) => gpu_helpers::upload_tensor(provider, &tensor)
            .map(gpu_helpers::resident_gpu_value)
            .map_err(|error| conv_error_with_detail(&CONV_ERROR_BUILD_OUTPUT, error)),
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(provider, &tensor)
            .map(gpu_helpers::complex_gpu_value),
        other => Ok(other),
    }
}

fn value_uses_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvMode {
    Full,
    Same,
    Valid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

#[derive(Clone, Copy, Debug)]
struct ConvInputMeta {
    len: usize,
    hint: OrientationHint,
}

#[derive(Clone)]
struct ConvInput {
    data: Vec<Complex<f64>>,
    len: usize,
    hint: OrientationHint,
}

fn parse_mode(args: &[Value]) -> BuiltinResult<ConvMode> {
    match args.len() {
        0 => Ok(ConvMode::Full),
        1 => {
            let Some(text) = tensor::value_to_string(&args[0]) else {
                return Err(conv_error(&CONV_ERROR_SHAPE_INVALID));
            };
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "full" => Ok(ConvMode::Full),
                "same" => Ok(ConvMode::Same),
                "valid" => Ok(ConvMode::Valid),
                _ => Err(conv_error(&CONV_ERROR_SHAPE_INVALID)),
            }
        }
        _ => Err(conv_error(&CONV_ERROR_ARG_COUNT)),
    }
}

fn try_conv_gpu(
    a: &Value,
    b: &Value,
    mode: ConvMode,
    single_output: bool,
) -> BuiltinResult<Option<Value>> {
    let (lhs_handle, rhs_handle) = match (a, b) {
        (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) => (lhs, rhs),
        _ => return Ok(None),
    };
    if runmat_accelerate_api::handle_integer_type(lhs_handle).is_some()
        || runmat_accelerate_api::handle_integer_type(rhs_handle).is_some()
        || runmat_accelerate_api::handle_is_logical(lhs_handle)
        || runmat_accelerate_api::handle_is_logical(rhs_handle)
        || value_is_complex(a)
        || value_is_complex(b)
    {
        return Ok(None);
    }
    if lhs_handle.device_id != rhs_handle.device_id {
        return Ok(None);
    }
    let provider = match runmat_accelerate_api::provider_for_handle(lhs_handle) {
        Some(p) => p,
        None => return Ok(None),
    };
    let Some(rhs_provider) = runmat_accelerate_api::provider_for_handle(rhs_handle) else {
        return Ok(None);
    };
    if !std::ptr::eq(provider, rhs_provider) {
        return Ok(None);
    }

    let lhs_meta = conv_meta_from_shape(&lhs_handle.shape);
    let rhs_meta = conv_meta_from_shape(&rhs_handle.shape);

    if lhs_meta.len == 0 || rhs_meta.len == 0 {
        return Ok(None);
    }

    let supported = |meta: &ConvInputMeta| {
        matches!(
            meta.hint,
            OrientationHint::Row | OrientationHint::Column | OrientationHint::Scalar
        )
    };

    if !supported(&lhs_meta) || !supported(&rhs_meta) {
        return Ok(None);
    }

    if matches!(mode, ConvMode::Valid) && lhs_meta.len < rhs_meta.len {
        return Ok(None);
    }

    let orientation = if matches!(mode, ConvMode::Full) {
        if matches!(lhs_meta.hint, OrientationHint::Column)
            && matches!(rhs_meta.hint, OrientationHint::Column)
        {
            Orientation::Column
        } else {
            Orientation::Row
        }
    } else {
        match lhs_meta.hint {
            OrientationHint::Column | OrientationHint::General => Orientation::Column,
            OrientationHint::Row => Orientation::Row,
            OrientationHint::Scalar | OrientationHint::Empty => {
                orientation_from_hints(lhs_meta.hint, rhs_meta.hint)
            }
        }
    };
    let provider_orientation = match orientation {
        Orientation::Row => ProviderConvOrientation::Row,
        Orientation::Column => ProviderConvOrientation::Column,
    };
    let provider_mode = match mode {
        ConvMode::Full => ProviderConvMode::Full,
        ConvMode::Same => ProviderConvMode::Same,
        ConvMode::Valid => ProviderConvMode::Valid,
    };

    let options = ProviderConv1dOptions {
        mode: provider_mode,
        orientation: provider_orientation,
    };

    let expected = if single_output {
        runmat_accelerate_api::ProviderPrecision::F32
    } else {
        runmat_accelerate_api::ProviderPrecision::F64
    };
    match provider.conv1d(lhs_handle, rhs_handle, options) {
        Ok(handle)
            if native_result_matches_provider(
                &handle,
                lhs_handle.device_id,
                provider,
                expected,
            ) =>
        {
            Ok(Some(Value::GpuTensor(handle)))
        }
        Ok(handle) => {
            free_rejected_native_handle(&handle, provider);
            Ok(None)
        }
        Err(err) => {
            log::trace!("conv: provider conv1d unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn native_result_matches_provider(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    input_device: u32,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_precision: runmat_accelerate_api::ProviderPrecision,
) -> bool {
    handle.device_id == input_device
        && runmat_accelerate_api::handle_precision(handle) == Some(expected_precision)
        && runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
}

fn free_rejected_native_handle(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
) {
    let owner = runmat_accelerate_api::provider_for_handle(handle).unwrap_or(invoked_provider);
    if let Err(err) = owner.free(handle) {
        log::trace!("conv: failed to free rejected provider result: {err}");
    }
}

async fn normalize_input(value: Value) -> BuiltinResult<ConvInput> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| {
                    let message = flow.message().to_owned();
                    conv_error_with_source(
                        &CONV_ERROR_GATHER_FAILED,
                        message,
                        map_control_flow_with_builtin(flow, BUILTIN_NAME),
                    )
                })?;
            normalize_host_input(gathered)
        }
        other => normalize_host_input(other),
    }
}

fn normalize_host_input(value: Value) -> BuiltinResult<ConvInput> {
    match value {
        Value::Tensor(tensor) => convert_tensor(tensor),
        Value::ComplexTensor(tensor) => convert_complex_tensor(tensor),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| conv_error_with_detail(&CONV_ERROR_CONVERSION, err))
            .and_then(convert_tensor),
        Value::Num(n) => Ok(ConvInput {
            data: vec![Complex::new(n, 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Int(i) => Ok(ConvInput {
            data: vec![Complex::new(i.to_f64(), 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Bool(b) => Ok(ConvInput {
            data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        Value::Complex(re, im) => Ok(ConvInput {
            data: vec![Complex::new(re, im)],
            len: 1,
            hint: OrientationHint::Scalar,
        }),
        other => Err(conv_error_with_detail(
            &CONV_ERROR_INVALID_INPUT,
            format!("{other:?}; expected numeric or logical values"),
        )),
    }
}

fn convert_tensor(tensor: Tensor) -> BuiltinResult<ConvInput> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let data = tensor::tensor_into_values_f64(tensor);
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    let data = data.into_iter().map(|re| Complex::new(re, 0.0)).collect();
    Ok(ConvInput { data, len, hint })
}

fn convert_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<ConvInput> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let data = tensor::complex_tensor_into_values_complex64(tensor);
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    Ok(ConvInput { data, len, hint })
}

fn classify_orientation(rows: usize, cols: usize, len: usize) -> OrientationHint {
    if rows == 1 && cols != 1 {
        OrientationHint::Row
    } else if cols == 1 && rows != 1 {
        OrientationHint::Column
    } else if rows == 1 && cols == 1 {
        OrientationHint::Scalar
    } else if len == 0 {
        OrientationHint::Empty
    } else {
        OrientationHint::General
    }
}

fn output_orientation(lhs: &ConvInput, rhs: &ConvInput, mode: ConvMode) -> Orientation {
    if matches!(mode, ConvMode::Full) {
        return if matches!(lhs.hint, OrientationHint::Column)
            && matches!(rhs.hint, OrientationHint::Column)
        {
            Orientation::Column
        } else {
            Orientation::Row
        };
    }
    match lhs.hint {
        OrientationHint::Column | OrientationHint::General => Orientation::Column,
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Scalar | OrientationHint::Empty => {
            orientation_from_hints(lhs.hint, rhs.hint)
        }
    }
}

fn orientation_from_hints(lhs: OrientationHint, rhs: OrientationHint) -> Orientation {
    match lhs {
        OrientationHint::Row => Orientation::Row,
        OrientationHint::Column => Orientation::Column,
        OrientationHint::General => Orientation::Column,
        OrientationHint::Scalar | OrientationHint::Empty => match rhs {
            OrientationHint::Column | OrientationHint::General => Orientation::Column,
            OrientationHint::Row => Orientation::Row,
            OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
        },
    }
}

fn conv_meta_from_shape(shape: &[usize]) -> ConvInputMeta {
    let len = tensor::element_count(shape);
    let (rows, cols) = shape_rows_cols(shape);
    let hint = classify_orientation(rows, cols, len);
    ConvInputMeta { len, hint }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        (0, 0)
    } else if shape.len() == 1 {
        (1, shape[0])
    } else {
        (shape[0], shape[1])
    }
}

fn convolve(a: &[Complex<f64>], b: &[Complex<f64>], single_output: bool) -> Vec<Complex<f64>> {
    if single_output {
        let mut out = vec![Complex::<f32>::new(0.0, 0.0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            let ai = Complex::<f32>::new(ai.re as f32, ai.im as f32);
            for (j, bj) in b.iter().enumerate() {
                let bj = Complex::<f32>::new(bj.re as f32, bj.im as f32);
                out[i + j] += ai * bj;
            }
        }
        return out
            .into_iter()
            .map(|value| Complex::new(f64::from(value.re), f64::from(value.im)))
            .collect();
    }
    let mut out = vec![Complex::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn apply_mode(
    full: Vec<Complex<f64>>,
    mode: ConvMode,
    len_a: usize,
    len_b: usize,
) -> Vec<Complex<f64>> {
    match mode {
        ConvMode::Full => full,
        ConvMode::Same => {
            if len_a == 0 {
                return Vec::new();
            }
            let start = len_b / 2;
            let end = (start + len_a).min(full.len());
            full[start..end].to_vec()
        }
        ConvMode::Valid => {
            if len_a == 0 || len_b == 0 || len_a < len_b {
                return Vec::new();
            }
            let start = len_b - 1;
            let valid_len = len_a - len_b + 1;
            let end = (start + valid_len).min(full.len());
            full[start..end].to_vec()
        }
    }
}

fn convert_output(
    data: Vec<Complex<f64>>,
    orientation: Orientation,
    single_output: bool,
    complex_output: bool,
) -> BuiltinResult<Value> {
    let len = data.len();
    let shape = match (orientation, len) {
        (Orientation::Row, 0) => vec![1, 0],
        (Orientation::Column, 0) => vec![0, 1],
        (Orientation::Row, _) => vec![1, len],
        (Orientation::Column, _) => vec![len, 1],
    };

    let all_real = data.iter().all(|c| c.im.abs() <= EPS);
    if all_real && !complex_output {
        if single_output {
            let real_data: Vec<f32> = data.into_iter().map(|c| c.re as f32).collect();
            let tensor = Tensor::from_f32(real_data, shape)
                .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_OUTPUT, &e))?;
            return Ok(tensor::tensor_into_value(tensor));
        }
        let real_data: Vec<f64> = data.into_iter().map(|c| c.re).collect();
        let tensor = Tensor::new(real_data, shape)
            .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_OUTPUT, &e))?;
        return Ok(tensor::tensor_into_value(tensor));
    }

    let storage = if single_output {
        ComplexStorage::F32(
            data.into_iter()
                .map(|c| (c.re as f32, c.im as f32))
                .collect(),
        )
    } else {
        ComplexStorage::F64(data.into_iter().map(|c| (c.re, c.im)).collect())
    };
    let tensor = ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| conv_error_with_detail(&CONV_ERROR_BUILD_COMPLEX_OUTPUT, &e))?;
    if !single_output && tensor.materialize_f64().len() == 1 {
        let (re, im) = tensor.materialize_f64()[0];
        return Ok(Value::Complex(re, im));
    }
    Ok(Value::ComplexTensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::{AccelProvider, HostTensorView};
    use runmat_builtins::{
        builtin_function_by_name, ComplexStorage, IntValue, IntegerStorage, LogicalArray,
        ResolveContext, Tensor, Type,
    };

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("typed integer tensor")
    }

    #[test]
    fn conv_type_full_uses_lengths() {
        let out = conv_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(4)])
            }
        );
    }

    #[test]
    fn conv_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("conv builtin");
        let descriptor = builtin.descriptor.expect("conv descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"C = conv(A, B)"));
        assert!(labels.contains(&"C = conv(A, B, shape)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.CONV.SHAPE_INVALID"));
        assert_eq!(CONV_INTEGER_CAPABILITIES.len(), 1);
        assert!(CONV_INTEGER_CAPABILITIES[0]
            .inputs
            .iter()
            .all(|input| input.classes.len() == 8));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_full_row_vectors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 6.0, 5.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_reads_typed_integer_vectors_exactly() {
        let a = integer_tensor(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]);
        let b = integer_tensor(IntegerStorage::U16(vec![1, 1, 1]), vec![1, 3]);

        let result = conv_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("conv");

        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 6.0, 5.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_accepts_all_integer_classes_and_returns_double() {
        let cases = [
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        for storage in cases {
            let input = integer_tensor(storage, vec![1, 2]);
            let output = conv_builtin(
                Value::Tensor(input),
                Value::Int(IntValue::U8(2)),
                Vec::new(),
            )
            .expect("integer conv");
            let Value::Tensor(output) = output else {
                panic!("expected double tensor")
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
            assert_eq!(output.materialize_f64(), vec![2.0, 4.0]);
        }
    }

    #[test]
    fn conv_full_orientation_uses_r2026a_both_columns_rule() {
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let col = Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(mixed) =
            conv_builtin(Value::Tensor(row), Value::Tensor(col.clone()), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(mixed.shape, vec![1, 3]);
        let Value::Tensor(columns) =
            conv_builtin(Value::Tensor(col.clone()), Value::Tensor(col), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(columns.shape, vec![3, 1]);
    }

    #[test]
    fn conv_typed_complex_integer_crosses_to_complex_double() {
        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![1, -1]),
        )
        .unwrap();
        let tensor = ComplexTensor::from_complex_storage(
            runmat_builtins::ComplexStorage::Integer(storage),
            vec![1, 2],
        )
        .unwrap();
        let output = conv_builtin(
            Value::ComplexTensor(tensor),
            Value::Int(IntValue::I8(2)),
            Vec::new(),
        )
        .expect("complex integer conv");
        let Value::ComplexTensor(output) = output else {
            panic!("complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F64);
        assert_eq!(output.materialize_f64(), vec![(2.0, 2.0), (4.0, -2.0)]);
    }

    #[test]
    fn conv_single_dominates_integer_output_class() {
        let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let integer = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let Value::Tensor(output) =
            conv_builtin(Value::Tensor(integer), Value::Tensor(single), Vec::new()).unwrap()
        else {
            panic!("tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
    }

    #[test]
    fn conv_resident_integer_gathers_exactly_and_restores_double_residency() {
        test_support::with_test_provider(|provider| {
            let input = integer_tensor(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![1, 2],
            );
            let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
            let output = conv_builtin(
                Value::GpuTensor(handle),
                Value::Int(IntValue::U8(1)),
                Vec::new(),
            )
            .unwrap();
            let Value::GpuTensor(handle) = output else {
                panic!("resident output")
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&handle), None);
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::Tensor(gathered) = gathered else {
                panic!("tensor")
            };
            assert_eq!(gathered.numeric_dtype(), runmat_builtins::NumericDType::F64);
            assert_eq!(
                gathered.materialize_f64(),
                vec![9_007_199_254_740_992.0, 18_446_744_073_709_551_616.0]
            );
        });
    }

    #[test]
    fn conv_complex_single_retains_class_and_explicit_complexity() {
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(1.0, 0.0), (2.0, 0.0)]),
            vec![1, 2],
        )
        .unwrap();
        let Value::ComplexTensor(output) = conv_builtin(
            Value::ComplexTensor(input),
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
            Vec::new(),
        )
        .unwrap() else {
            panic!("explicitly complex tensor")
        };
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(output.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
    }

    #[test]
    fn conv_resident_complex_gathers_computes_and_restores() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -1.0)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
            let output = conv_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("resident complex conv");
            let Value::GpuTensor(handle) = output else {
                panic!("resident complex output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let Value::ComplexTensor(gathered) =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap()
            else {
                panic!("complex tensor")
            };
            assert_eq!(gathered.materialize_f64(), vec![(2.0, 4.0), (6.0, -2.0)]);
        });
    }

    #[test]
    fn conv_mixed_provider_inputs_gather_independently() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let a = provider_a
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        let b = provider_b
            .upload(&HostTensorView {
                data: &[1.0, 1.0],
                shape: &[1, 2],
            })
            .unwrap();
        let Value::GpuTensor(output) =
            conv_builtin(Value::GpuTensor(a), Value::GpuTensor(b), Vec::new()).unwrap()
        else {
            panic!("owner-restored output")
        };
        assert_eq!(output.device_id, provider_a.device_id());
        let gathered = block_on(provider_a.download(&output)).unwrap();
        assert_eq!(gathered.data, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn conv_rejects_native_output_with_wrong_single_precision() {
        test_support::with_test_provider(|provider| {
            let a = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let b = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .unwrap();
            runmat_accelerate_api::set_handle_precision(
                &a,
                runmat_accelerate_api::ProviderPrecision::F32,
            );
            let Value::Tensor(output) =
                conv_builtin(Value::GpuTensor(a), Value::GpuTensor(b), Vec::new()).unwrap()
            else {
                panic!("class-preserving host fallback")
            };
            assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
            assert_eq!(output.materialize_f64(), vec![1.0, 3.0, 2.0]);
        });
    }

    #[test]
    fn conv_rejects_native_result_owned_by_another_provider() {
        let _guard = test_support::accel_test_lock();
        let provider_a: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let provider_b: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(provider_a);
            runmat_accelerate_api::register_provider(provider_b);
        }
        let wrong_owner = provider_b
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        assert!(!native_result_matches_provider(
            &wrong_owner,
            provider_a.device_id(),
            provider_a,
            runmat_accelerate_api::ProviderPrecision::F64,
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_same_matches_length() {
        let a = Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![1, 5]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b.clone()),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let result_column = conv_builtin(
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![5, 1]).unwrap()),
            Value::Tensor(b),
            vec![Value::from("same")],
        )
        .expect("conv");
        match result_column {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 1]);
                assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 2.0, 2.0, -6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_same_even_kernel_uses_matlab_alignment() {
        let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = conv_builtin(
            Value::Tensor(signal),
            Value::Tensor(kernel),
            vec![Value::from("same")],
        )
        .expect("conv");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(result.shape, vec![1, 5]);
        assert_eq!(result.materialize_f64(), vec![10.0, 20.0, 30.0, 34.0, 31.0]);
    }

    #[test]
    fn conv_frees_rejected_native_result_before_fallback() {
        test_support::with_rejecting_native_result_provider(|provider| {
            let signal = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .unwrap();
            let kernel = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .unwrap();
            let result = conv_builtin(
                Value::GpuTensor(signal),
                Value::GpuTensor(kernel),
                Vec::new(),
            )
            .expect("conv fallback");
            let result = test_support::gather(result).expect("gather fallback");
            assert_eq!(result.materialize_f64(), vec![1.0, 3.0, 5.0, 3.0]);
            assert_eq!(provider.free_count(), 1);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_valid_empty_when_kernel_longer() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let result = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn conv_valid_empty_second_vector_returns_zeros_like_first() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::from_f32(Vec::new(), vec![0, 1]).unwrap();
        let Value::Tensor(output) = conv_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("valid")],
        )
        .unwrap() else {
            panic!("tensor")
        };
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(output.numeric_dtype(), runmat_builtins::NumericDType::F32);
        assert_eq!(output.materialize_f64(), vec![0.0, 0.0, 0.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_complex_inputs() {
        let a = Value::Complex(1.0, 2.0);
        let b = Value::Complex(3.0, -1.0);
        let result = conv_builtin(a, b, Vec::new()).expect("conv");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 5.0).abs() < 1e-12);
                assert!((im - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_scalar_times_vector_follows_other_orientation() {
        let scalar = Value::Num(2.0);
        let vec = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let result = conv_builtin(scalar, Value::Tensor(vec), Vec::new()).expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![8.0, 10.0, 12.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_handles_empty_inputs_with_row_orientation() {
        let empty_row = Tensor::new(Vec::<f64>::new(), vec![1, 0]).unwrap();
        let kernel = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_row), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_handles_empty_inputs_with_column_orientation() {
        let empty_col = Tensor::new(Vec::<f64>::new(), vec![0, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            conv_builtin(Value::Tensor(empty_col), Value::Tensor(kernel), Vec::new()).unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_promotes_logical_inputs_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let kernel = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::LogicalArray(logical.clone()),
            Value::Tensor(kernel),
            Vec::new(),
        )
        .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        // ensure logical inputs on RHS follow the same promotion path
        let lhs = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(Value::Tensor(lhs), Value::LogicalArray(logical), Vec::new())
            .expect("conv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_rejects_invalid_shape_keyword() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = error_message(
            conv_builtin(
                Value::Tensor(a),
                Value::Tensor(b),
                vec![Value::from("diagonal")],
            )
            .unwrap_err(),
        );
        assert!(err.contains("third argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_rejects_non_numeric_input() {
        let err = error_message(
            conv_builtin(Value::from("hi"), Value::Num(1.0), Vec::new()).unwrap_err(),
        );
        assert!(err.contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let kernel = Tensor::new(vec![1.0, 0.0, -1.0], vec![1, 3]).unwrap();
            let host_expected = conv_builtin(
                Value::Tensor(signal.clone()),
                Value::Tensor(kernel.clone()),
                Vec::new(),
            )
            .expect("host conv");

            let sig_view = HostTensorView {
                data: &signal.materialize_f64(),
                shape: &signal.shape,
            };
            let ker_view = HostTensorView {
                data: &kernel.materialize_f64(),
                shape: &kernel.shape,
            };
            let sig_handle = provider.upload(&sig_view).expect("upload signal");
            let ker_handle = provider.upload(&ker_view).expect("upload kernel");
            let gpu_result = conv_builtin(
                Value::GpuTensor(sig_handle),
                Value::GpuTensor(ker_handle),
                Vec::new(),
            )
            .expect("gpu conv");

            let gathered = test_support::gather(gpu_result).expect("gather gpu");
            let expected = test_support::gather(host_expected).expect("gather host");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn conv_wgpu_matches_cpu_same_mode() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("provider registry");
        let signal = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 5]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();

        let sig_view = HostTensorView {
            data: &signal.materialize_f64(),
            shape: &signal.shape,
        };
        let ker_view = HostTensorView {
            data: &kernel.materialize_f64(),
            shape: &kernel.shape,
        };
        let sig_handle = provider.upload(&sig_view).expect("upload signal");
        let ker_handle = provider.upload(&ker_view).expect("upload kernel");

        let gpu_value = conv_builtin(
            Value::GpuTensor(sig_handle),
            Value::GpuTensor(ker_handle),
            vec![Value::from("same")],
        )
        .expect("gpu conv");

        let gathered_gpu = test_support::gather(gpu_value).expect("gather gpu");
        assert_eq!(gathered_gpu.shape, vec![1, 5]);
        assert_eq!(
            gathered_gpu.materialize_f64(),
            vec![10.0, 20.0, 30.0, 34.0, 31.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conv_same_with_integer_dimension_argument() {
        let signal = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let kernel = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = conv_builtin(
            Value::Tensor(signal),
            Value::Tensor(kernel),
            vec![Value::Int(IntValue::I32(1))],
        );
        assert!(result.is_err());
    }

    fn conv_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::conv_builtin(a, b, rest))
    }
}
