//! MATLAB-compatible `pulstran` builtin for sampled pulse trains.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::gauspuls::{
    default_params as gauspuls_default_params, gauspuls_scalar, validate_params, GauspulsParams,
};
use crate::builtins::math::signal::rectpuls::{
    rectpuls_scalar, validate_width as validate_rect_width,
};
use crate::builtins::math::signal::tripuls::{
    tripuls_scalar, validate_skew as validate_tripuls_skew,
    validate_width as validate_tripuls_width,
};
use crate::builtins::math::signal::type_resolvers::pulse_train_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "pulstran";

const PULSTRAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pulse train sampled at T.",
}];

const PULSTRAN_INPUTS_FUNCTION: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "FUN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse function name or function handle.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional pulse function parameters.",
    },
];

const PULSTRAN_INPUTS_PROTO: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pulse delays, or an N-by-2 delay/amplitude matrix.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sampled prototype pulse.",
    },
    BuiltinParamDescriptor {
        name: "FS",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Prototype sample rate.",
    },
];

const PULSTRAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, FUN, P1, ...)",
        inputs: &PULSTRAN_INPUTS_FUNCTION,
        outputs: &PULSTRAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pulstran(T, D, P, FS)",
        inputs: &PULSTRAN_INPUTS_PROTO,
        outputs: &PULSTRAN_OUTPUT,
    },
];

const PULSTRAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_INPUT",
    identifier: Some("RunMat:pulstran:InvalidInput"),
    when: "T, D, or sampled prototype inputs are not real numeric arrays.",
    message: "pulstran: expected real numeric input",
};

const PULSTRAN_ERROR_INVALID_DELAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_DELAY",
    identifier: Some("RunMat:pulstran:InvalidDelay"),
    when: "Delay input is not a vector or N-by-2 delay/amplitude matrix.",
    message: "pulstran: D must be a delay vector or N-by-2 delay/amplitude matrix",
};

const PULSTRAN_ERROR_INVALID_PULSE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PULSE",
    identifier: Some("RunMat:pulstran:InvalidPulse"),
    when: "Pulse function or sampled prototype is malformed.",
    message: "pulstran: invalid pulse specification",
};

const PULSTRAN_ERROR_INVALID_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INVALID_PARAMETER",
    identifier: Some("RunMat:pulstran:InvalidParameter"),
    when: "Pulse function parameters or sampled prototype rate are malformed.",
    message: "pulstran: invalid pulse parameter",
};

const PULSTRAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.ARG_COUNT",
    identifier: Some("RunMat:pulstran:ArgCount"),
    when: "Required arguments are missing or too many prototype arguments are provided.",
    message: "pulstran: expected pulstran(T, D, FUN, ...) or pulstran(T, D, P, FS)",
};

const PULSTRAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PULSTRAN.INTERNAL",
    identifier: Some("RunMat:pulstran:InternalError"),
    when: "Internal tensor construction, callback result materialization, or GPU gather fails.",
    message: "pulstran: internal error",
};

const PULSTRAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    PULSTRAN_ERROR_INVALID_INPUT,
    PULSTRAN_ERROR_INVALID_DELAY,
    PULSTRAN_ERROR_INVALID_PULSE,
    PULSTRAN_ERROR_INVALID_PARAMETER,
    PULSTRAN_ERROR_ARG_COUNT,
    PULSTRAN_ERROR_INTERNAL,
];

pub const PULSTRAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PULSTRAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PULSTRAN_ERRORS,
};

const PULSTRAN_INTEGER_T_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerTimeExtension"),
};
const PULSTRAN_INTEGER_D_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-delay",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with typed-integer delays or gains is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerDelayExtension"),
};
const PULSTRAN_INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-integer-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with a typed-integer sampled prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranIntegerPrototypeExtension"),
};
const PULSTRAN_INTEGER_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pulstran-integer-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pulstran with typed-integer built-in pulse parameters is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PulstranIntegerParameterExtension"),
    };
const PULSTRAN_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pulstran-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pulstran with an explicit gpuArray computation input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PulstranExplicitGpuInputExtension"),
};
pub const PULSTRAN_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    PULSTRAN_INTEGER_T_EXTENSION,
    PULSTRAN_INTEGER_D_EXTENSION,
    PULSTRAN_INTEGER_P_EXTENSION,
    PULSTRAN_INTEGER_PARAMETER_EXTENSION,
    PULSTRAN_EXPLICIT_GPU_EXTENSION,
];

const PULSTRAN_INTEGER_T_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public page does not document typed-integer sample times; RunMat admits them only at an exact binary64 waveform boundary.",
    }];
const PULSTRAN_INTEGER_D_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "D",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "Typed delays and optional gains are independently gated before waveform evaluation.",
    }];
const PULSTRAN_INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "P/FS/built-in pulse parameters",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed sampled prototypes and numeric controls cross checked floating interpolation or pulse-generation boundaries; callback extra arguments remain exact pass-through values.",
    }];
pub const PULSTRAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(integer_T,D,FUN,...) or pulstran(integer_T,D,P,FS)",
        inputs: &PULSTRAN_INTEGER_T_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sample times are checked before provider access and converted once for waveform evaluation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(T,integer_D,FUN,...) or pulstran(T,integer_D,P,FS)",
        inputs: &PULSTRAN_INTEGER_D_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Delay and gain values are checked independently before summation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = pulstran(T,D,integer_P,integer_FS) or built-in pulse with integer parameters",
        inputs: &PULSTRAN_INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed numeric role is admitted separately; function-handle callback payloads are not reclassified as pulstran computation controls.",
    },
];

#[derive(Clone, Copy, Debug)]
struct PulseInstance {
    delay: f64,
    amplitude: f64,
}

#[derive(Clone, Debug)]
enum PulseSource {
    Rect { width: f64 },
    Tri { width: f64, skew: f64 },
    Gaus { params: GauspulsParams },
    Callback { handle: Value, args: Vec<Value> },
    Prototype { samples: Vec<f64>, fs: f64 },
}

fn pulstran_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    pulstran_error_with_message(error.message, error)
}

fn pulstran_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    pulstran_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn pulstran_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pulstran_error_with_source(
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
    name = "pulstran",
    category = "math/signal",
    summary = "Generate pulse trains from pulse functions or sampled prototypes.",
    keywords = "pulstran,pulse train,rectpuls,tripuls,gauspuls,signal processing",
    type_resolver(pulse_train_type),
    descriptor(crate::builtins::math::signal::pulstran::PULSTRAN_DESCRIPTOR),
    extensions(crate::builtins::math::signal::pulstran::PULSTRAN_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::pulstran::PULSTRAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::pulstran"
)]
async fn pulstran_builtin(
    t: Value,
    d: Value,
    pulse: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    ensure_pulstran_extensions(&t, &d, &pulse, &rest).await?;
    let gpu_source = pulstran_gpu_source(&t, &d, &pulse, &rest)?;
    let t = real_tensor_arg(t, &PULSTRAN_ERROR_INVALID_INPUT).await?;
    let delays = parse_delays(real_tensor_arg(d, &PULSTRAN_ERROR_INVALID_DELAY).await?)?;
    let source = parse_pulse_source(pulse, rest).await?;
    let y = evaluate_pulse_train(&t, &delays, &source).await?;
    let Some(source) = gpu_source else {
        return Ok(tensor_into_value(y));
    };
    let restored =
        gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(y), BUILTIN_NAME)?;
    if runmat_accelerate_api::handle_is_explicit(&source)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INTERNAL,
            "provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

async fn ensure_pulstran_extensions(
    t: &Value,
    d: &Value,
    pulse: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    crate::builtins::common::validation::reject_typed_complex_integer(t, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(d, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        t,
        &PULSTRAN_INTEGER_T_EXTENSION,
        BUILTIN_NAME,
        "T",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        d,
        &PULSTRAN_INTEGER_D_EXTENSION,
        BUILTIN_NAME,
        "D",
    )
    .await?;
    let sampled = is_numeric_or_gpu(pulse);
    if sampled && crate::builtins::common::validation::value_has_native_integer_class(pulse) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_INTEGER_P_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if (sampled || pulse_uses_builtin_parameters(pulse))
        && rest
            .iter()
            .any(|value| crate::builtins::common::validation::value_has_native_integer_class(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if pulstran_resident_values(t, d, pulse, rest).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &PULSTRAN_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn pulse_uses_builtin_parameters(pulse: &Value) -> bool {
    let name = text_scalar(pulse).or_else(|| match pulse {
        Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => Some(name.clone()),
        _ => None,
    });
    name.is_some_and(|name| {
        matches!(
            name.trim().to_ascii_lowercase().as_str(),
            "rectpuls" | "tripuls" | "gauspuls"
        )
    })
}

fn pulstran_resident_values<'a>(
    t: &'a Value,
    d: &'a Value,
    pulse: &'a Value,
    rest: &'a [Value],
) -> impl Iterator<Item = &'a Value> {
    let uses_controls = is_numeric_or_gpu(pulse) || pulse_uses_builtin_parameters(pulse);
    std::iter::once(t)
        .chain(std::iter::once(d))
        .chain(std::iter::once(pulse).filter(|value| is_numeric_or_gpu(value)))
        .chain(rest.iter().filter(move |_| uses_controls))
}

fn pulstran_gpu_source(
    t: &Value,
    d: &Value,
    pulse: &Value,
    rest: &[Value],
) -> BuiltinResult<Option<runmat_accelerate_api::GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        pulstran_resident_values(t, d, pulse, rest).filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        BUILTIN_NAME,
    )
}

async fn real_tensor_arg(
    value: Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    pulstran_error_with_source(
                        &PULSTRAN_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(pulstran_error(error)),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(pulstran_error(error))
        }
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| pulstran_error_with_detail(error, err)),
    }
}

fn parse_delays(tensor: Tensor) -> BuiltinResult<Vec<PulseInstance>> {
    if tensor.is_empty() {
        return Ok(Vec::new());
    }
    let values = tensor::tensor_values_f64(&tensor);
    let shape = tensor.shape.as_slice();
    if shape.len() <= 2 {
        let rows = shape.first().copied().unwrap_or(values.len());
        let cols = shape.get(1).copied().unwrap_or(1);
        let is_vector = rows == 1 || cols == 1 || values.len() == 1;
        if !is_vector && cols == 2 {
            return Ok((0..rows)
                .map(|row| PulseInstance {
                    delay: values[row],
                    amplitude: values[row + rows],
                })
                .collect());
        }
        if is_vector {
            return Ok(values
                .into_iter()
                .map(|delay| PulseInstance {
                    delay,
                    amplitude: 1.0,
                })
                .collect());
        }
    }
    Err(pulstran_error(&PULSTRAN_ERROR_INVALID_DELAY))
}

async fn parse_pulse_source(pulse: Value, rest: Vec<Value>) -> BuiltinResult<PulseSource> {
    if is_numeric_or_gpu(&pulse) {
        return parse_sampled_prototype(pulse, rest).await;
    }
    if let Some(name) = text_scalar(&pulse) {
        let handle_name = normalize_pulse_name(&name)?;
        return parse_named_pulse_or_callback(handle_name, rest).await;
    }
    match pulse {
        Value::FunctionHandle(name) | Value::ExternalFunctionHandle(name) => {
            let handle_name = normalize_pulse_name(&name)?;
            parse_named_pulse_or_callback(handle_name, rest).await
        }
        Value::BoundFunctionHandle { .. } | Value::MethodFunctionHandle(_) => {
            Ok(PulseSource::Callback {
                handle: pulse,
                args: rest,
            })
        }
        _ => Err(pulstran_error(&PULSTRAN_ERROR_INVALID_PULSE)),
    }
}

async fn parse_named_pulse_or_callback(
    name: String,
    rest: Vec<Value>,
) -> BuiltinResult<PulseSource> {
    match name.to_ascii_lowercase().as_str() {
        "rectpuls" => Ok(PulseSource::Rect {
            width: parse_rect_width(&rest).await?,
        }),
        "tripuls" => {
            let (width, skew) = parse_tripuls_options(&rest).await?;
            Ok(PulseSource::Tri { width, skew })
        }
        "gauspuls" => Ok(PulseSource::Gaus {
            params: parse_gauspuls_params(&rest).await?,
        }),
        _ => Ok(PulseSource::Callback {
            handle: Value::FunctionHandle(name),
            args: rest,
        }),
    }
}

async fn parse_sampled_prototype(pulse: Value, rest: Vec<Value>) -> BuiltinResult<PulseSource> {
    if rest.len() > 1 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 3),
        ));
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&pulse, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &pulse,
        &PULSTRAN_INTEGER_P_EXTENSION,
        BUILTIN_NAME,
        "prototype",
    )
    .await?;
    let prototype = real_tensor_arg(pulse, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    if !is_vector_shape(&prototype.shape) {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "sampled prototype must be a vector",
        ));
    }
    let fs = match rest.first() {
        Some(value) => {
            crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
                BUILTIN_NAME,
                "FS",
            )
            .await?;
            let raw = scalar_f64_from_value_async(value)
                .await
                .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?
                .ok_or_else(|| pulstran_error(&PULSTRAN_ERROR_INVALID_PARAMETER))?;
            if !raw.is_finite() || raw <= 0.0 {
                return Err(pulstran_error_with_detail(
                    &PULSTRAN_ERROR_INVALID_PARAMETER,
                    format!("sample rate must be positive and finite, got {raw}"),
                ));
            }
            raw
        }
        None => 1.0,
    };
    Ok(PulseSource::Prototype {
        samples: tensor::tensor_into_values_f64(prototype),
        fs,
    })
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

fn is_numeric_or_gpu(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

fn normalize_pulse_name(name: &str) -> BuiltinResult<String> {
    let trimmed = name.trim();
    let trimmed = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    if trimmed.is_empty() {
        Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            "pulse function name must not be empty",
        ))
    } else {
        Ok(trimmed.to_string())
    }
}

async fn parse_rect_width(rest: &[Value]) -> BuiltinResult<f64> {
    match rest.len() {
        0 => Ok(1.0),
        1 => {
            let raw = scalar_arg(&rest[0], "width").await?;
            validate_rect_width(raw)
                .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))
        }
        _ => Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("rectpuls got {}", rest.len() + 1),
        )),
    }
}

async fn parse_tripuls_options(rest: &[Value]) -> BuiltinResult<(f64, f64)> {
    if rest.len() > 2 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("tripuls got {}", rest.len() + 1),
        ));
    }
    let width = match rest.first() {
        Some(value) => validate_tripuls_width(scalar_arg(value, "width").await?)
            .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?,
        None => 1.0,
    };
    let skew = match rest.get(1) {
        Some(value) => validate_tripuls_skew(scalar_arg(value, "skew").await?)
            .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))?,
        None => 0.0,
    };
    Ok((width, skew))
}

async fn parse_gauspuls_params(rest: &[Value]) -> BuiltinResult<GauspulsParams> {
    if rest.len() > 3 {
        return Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_ARG_COUNT,
            format!("gauspuls got {}", rest.len() + 1),
        ));
    }
    let mut params = gauspuls_default_params();
    if let Some(value) = rest.first() {
        params.fc = scalar_arg(value, "FC").await?;
    }
    if let Some(value) = rest.get(1) {
        params.bw = scalar_arg(value, "BW").await?;
    }
    if let Some(value) = rest.get(2) {
        params.bwr = scalar_arg(value, "BWR").await?;
    }
    validate_params(params)
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, err))
}

async fn scalar_arg(value: &Value, label: &str) -> BuiltinResult<f64> {
    crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &PULSTRAN_INTEGER_PARAMETER_EXTENSION,
        BUILTIN_NAME,
        label,
    )
    .await?;
    scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            pulstran_error_with_detail(&PULSTRAN_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?
        .ok_or_else(|| {
            pulstran_error_with_detail(
                &PULSTRAN_ERROR_INVALID_PARAMETER,
                format!("{label}: expected scalar"),
            )
        })
}

async fn evaluate_pulse_train(
    t: &Tensor,
    delays: &[PulseInstance],
    source: &PulseSource,
) -> BuiltinResult<Tensor> {
    let times = tensor::tensor_values_f64_cow(t);
    let mut out = vec![0.0; times.len()];
    for pulse in delays {
        match source {
            PulseSource::Rect { width } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * rectpuls_scalar(time - pulse.delay, *width);
                }
            }
            PulseSource::Tri { width, skew } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * tripuls_scalar(time - pulse.delay, *width, *skew);
                }
            }
            PulseSource::Gaus { params } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * gauspuls_scalar(time - pulse.delay, *params);
                }
            }
            PulseSource::Prototype { samples, fs } => {
                for (idx, &time) in times.iter().enumerate() {
                    out[idx] += pulse.amplitude * prototype_value(samples, *fs, time - pulse.delay);
                }
            }
            PulseSource::Callback { handle, args } => {
                let shifted = shifted_time_value(t, pulse.delay)?;
                let mut call_args = Vec::with_capacity(args.len() + 1);
                call_args.push(shifted);
                call_args.extend(args.iter().cloned());
                let value = crate::call_feval_async_with_outputs(handle.clone(), &call_args, 1)
                    .await
                    .map_err(|err| {
                        pulstran_error_with_source(
                            &PULSTRAN_ERROR_INVALID_PULSE,
                            "pulse function call failed",
                            err,
                        )
                    })?;
                let samples = callback_samples(value, times.len()).await?;
                for (idx, sample) in samples.into_iter().enumerate() {
                    out[idx] += pulse.amplitude * sample;
                }
            }
        }
    }
    Tensor::new(out, t.shape.clone())
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

fn shifted_time_value(t: &Tensor, delay: f64) -> BuiltinResult<Value> {
    let data = tensor::tensor_values_f64_cow(t)
        .iter()
        .map(|value| value - delay)
        .collect::<Vec<_>>();
    Tensor::new(data, t.shape.clone())
        .map(tensor_into_value)
        .map_err(|err| pulstran_error_with_detail(&PULSTRAN_ERROR_INTERNAL, &err))
}

async fn callback_samples(value: Value, expected_len: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = real_tensor_arg(value, &PULSTRAN_ERROR_INVALID_PULSE).await?;
    let actual_len = tensor.len();
    if actual_len == expected_len {
        Ok(tensor::tensor_into_values_f64(tensor))
    } else if actual_len == 1 {
        Ok(vec![tensor::tensor_value_f64(&tensor, 0); expected_len])
    } else {
        Err(pulstran_error_with_detail(
            &PULSTRAN_ERROR_INVALID_PULSE,
            format!(
                "pulse function returned {} samples for {expected_len} input samples",
                actual_len
            ),
        ))
    }
}

fn prototype_value(samples: &[f64], fs: f64, t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if !t.is_finite() || samples.is_empty() {
        return 0.0;
    }
    let x = t * fs;
    if x < 0.0 || x > (samples.len() - 1) as f64 {
        return 0.0;
    }
    if samples.len() == 1 {
        return if x == 0.0 { samples[0] } else { 0.0 };
    }
    let lower = x.floor() as usize;
    if lower + 1 >= samples.len() {
        return samples[lower];
    }
    let frac = x - lower as f64;
    samples[lower] * (1.0 - frac) + samples[lower + 1] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, CharArray, IntegerStorage, StringArray};

    fn call(t: Value, d: Value, pulse: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(pulstran_builtin(t, d, pulse, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
    }

    #[cfg(feature = "wgpu")]
    fn all_integer_storages() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![0, 1]),
            IntegerStorage::I16(vec![0, 1]),
            IntegerStorage::I32(vec![0, 1]),
            IntegerStorage::I64(vec![0, 1]),
            IntegerStorage::U8(vec![0, 1]),
            IntegerStorage::U16(vec![0, 1]),
            IntegerStorage::U32(vec![0, 1]),
            IntegerStorage::U64(vec![0, 1]),
        ]
    }

    #[test]
    fn pulstran_rectpuls_named_char_reproduces_pulse_train() {
        let t = Tensor::new(vec![-0.5, 0.0, 0.5, 1.0], vec![1, 4]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let pulse = Value::CharArray(CharArray::new_row("rectpuls"));
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn pulstran_accepts_delay_amplitude_matrix() {
        let t = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let pulse =
            Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
    }

    #[test]
    fn pulstran_reads_typed_integer_times_and_delay_amplitudes_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let t = integer_tensor(vec![0, 1], vec![1, 2]);
        let d = integer_tensor(vec![0, 1, 2, 3], vec![2, 2]);
        let pulse =
            Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
    }

    #[test]
    fn pulstran_reads_native_single_times_and_delays_authoritatively() {
        let t = Tensor::from_f32(vec![0.0, 1.0], vec![1, 2]).expect("single times");
        let d = Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2])
            .expect("single delay amplitudes");
        let pulse =
            Value::StringArray(StringArray::new(vec!["rectpuls".to_string()], vec![1, 1]).unwrap());
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                pulse,
                vec![Value::Num(0.25)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.materialize_f64(), vec![2.0, 3.0]);
    }

    #[test]
    fn pulstran_sampled_prototype_interpolates_at_sample_rate() {
        let t = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let prototype = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                Value::Tensor(prototype),
                vec![Value::Num(2.0)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn pulstran_reads_typed_integer_sampled_prototype_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let t = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![1, 5]).unwrap();
        let d = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let prototype = integer_tensor(vec![0, 1, 0], vec![1, 3]);
        let out = expect_tensor(
            call(
                Value::Tensor(t),
                Value::Tensor(d),
                Value::Tensor(prototype),
                vec![Value::Num(2.0)],
            )
            .expect("pulstran"),
        );
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn pulstran_callback_samples_read_typed_integer_storage_exactly() {
        let samples = block_on(callback_samples(
            Value::Tensor(integer_tensor(vec![2, 4, 6], vec![1, 3])),
            3,
        ))
        .expect("callback samples");
        assert_eq!(samples, vec![2.0, 4.0, 6.0]);

        let repeated = block_on(callback_samples(
            Value::Tensor(integer_tensor(vec![7], vec![1, 1])),
            3,
        ))
        .expect("scalar callback sample");
        assert_eq!(repeated, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn pulstran_rejects_invalid_delay_shape_and_bad_sample_rate() {
        let t = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
        let bad_d = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2, 1]).unwrap());
        let err = call(
            t.clone(),
            bad_d,
            Value::CharArray(CharArray::new_row("rectpuls")),
            Vec::new(),
        )
        .expect_err("delay shape");
        assert_eq!(err.identifier(), PULSTRAN_ERROR_INVALID_DELAY.identifier);

        let d = Value::Num(0.0);
        let prototype = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        let err = call(t, d, prototype, vec![Value::Num(0.0)]).expect_err("sample rate");
        assert_eq!(
            err.identifier(),
            PULSTRAN_ERROR_INVALID_PARAMETER.identifier
        );
    }

    #[test]
    fn pulstran_automatic_residency_is_transparent_and_explicit_residency_is_gated() {
        use crate::builtins::common::test_support;

        test_support::with_test_provider(|provider| {
            let times = Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("times");
            let automatic = gpu_helpers::upload_tensor(provider, &times).expect("upload");
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let output = call(
                Value::GpuTensor(automatic),
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![Value::Num(0.25)],
            )
            .expect("automatic resident input");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected automatic resident output");
            };
            assert!(!runmat_accelerate_api::handle_is_explicit(output_handle));
            assert_eq!(
                test_support::gather(output)
                    .expect("gather")
                    .materialize_f64(),
                vec![1.0, 0.0]
            );

            let explicit = gpu_helpers::upload_tensor(provider, &times).expect("upload");
            runmat_accelerate_api::mark_handle_explicit(&explicit);
            let error = call(
                Value::GpuTensor(explicit),
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![Value::Num(0.25)],
            )
            .expect_err("strict explicit input");
            assert_eq!(
                error.identifier(),
                PULSTRAN_EXPLICIT_GPU_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn pulstran_wgpu_fallback_enforces_double_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_integer_storages() {
            let times = Tensor::new_integer(storage, vec![1, 2]).expect("integer times");
            let handle = gpu_helpers::upload_tensor(provider, &times).expect("upload");
            runmat_accelerate_api::mark_handle_explicit(&handle);
            let result = call(
                Value::GpuTensor(handle),
                Value::Num(0.0),
                Value::from("rectpuls"),
                vec![Value::Num(0.25)],
            );
            if provider.precision() == runmat_accelerate_api::ProviderPrecision::F64 {
                let output = result.expect("resident integer pulstran");
                let Value::GpuTensor(output_handle) = &output else {
                    panic!("expected resident output");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
                assert_eq!(
                    test_support::gather(output)
                        .expect("gather")
                        .materialize_f64(),
                    vec![1.0, 0.0]
                );
            } else {
                let error = result.expect_err("f32 owner cannot preserve double output");
                assert!(error
                    .message()
                    .contains("cannot preserve explicit gpuArray"));
            }
        }
    }

    #[test]
    fn pulstran_is_registered() {
        assert!(builtin_function_by_name("pulstran").is_some());
    }
}
