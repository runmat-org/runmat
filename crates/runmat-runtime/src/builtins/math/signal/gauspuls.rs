//! MATLAB-compatible `gauspuls` builtin for Gaussian-modulated sinusoid samples.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::signal::type_resolvers::numeric_unary_shape_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gauspuls";
const DEFAULT_FC: f64 = 1000.0;
const DEFAULT_BW: f64 = 0.5;
const DEFAULT_BWR: f64 = -6.0;
const DEFAULT_TPE: f64 = -60.0;

const GAUSPULS_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "YI",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "In-phase Gaussian pulse samples.",
    },
    BuiltinParamDescriptor {
        name: "YQ",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Quadrature Gaussian pulse samples.",
    },
    BuiltinParamDescriptor {
        name: "YE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Gaussian pulse envelope samples.",
    },
];

const GAUSPULS_CUTOFF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TC",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cutoff time where the envelope reaches TPE dB.",
}];

const GAUSPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const GAUSPULS_INPUTS_T_FC_BW_BWR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "FC",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1000"),
        description: "Carrier frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "BW",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.5"),
        description: "Fractional bandwidth.",
    },
    BuiltinParamDescriptor {
        name: "BWR",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-6"),
        description: "Bandwidth reference level in dB.",
    },
];

const GAUSPULS_INPUTS_CUTOFF: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`\"cutoff\"` requests cutoff time.",
    },
    BuiltinParamDescriptor {
        name: "FC",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1000"),
        description: "Carrier frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "BW",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.5"),
        description: "Fractional bandwidth.",
    },
    BuiltinParamDescriptor {
        name: "BWR",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-6"),
        description: "Bandwidth reference level in dB.",
    },
    BuiltinParamDescriptor {
        name: "TPE",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-60"),
        description: "Trailing pulse envelope level in dB.",
    },
];

const GAUSPULS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = gauspuls(T)",
        inputs: &GAUSPULS_INPUTS_T,
        outputs: &GAUSPULS_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[YI, YQ, YE] = gauspuls(T, FC, BW, BWR)",
        inputs: &GAUSPULS_INPUTS_T_FC_BW_BWR,
        outputs: &GAUSPULS_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "TC = gauspuls(\"cutoff\", FC, BW, BWR, TPE)",
        inputs: &GAUSPULS_INPUTS_CUTOFF,
        outputs: &GAUSPULS_CUTOFF_OUTPUT,
    },
];

const GAUSPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INVALID_INPUT",
    identifier: Some("RunMat:gauspuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "gauspuls: expected real numeric input",
};

const GAUSPULS_ERROR_INVALID_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INVALID_PARAMETER",
    identifier: Some("RunMat:gauspuls:InvalidParameter"),
    when: "Frequency, bandwidth, reference level, or cutoff level is malformed.",
    message: "gauspuls: invalid parameter",
};

const GAUSPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.ARG_COUNT",
    identifier: Some("RunMat:gauspuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message:
        "gauspuls: expected gauspuls(T, FC, BW, BWR) or gauspuls(\"cutoff\", FC, BW, BWR, TPE)",
};

const GAUSPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INTERNAL",
    identifier: Some("RunMat:gauspuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "gauspuls: internal error",
};

const GAUSPULS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    GAUSPULS_ERROR_INVALID_INPUT,
    GAUSPULS_ERROR_INVALID_PARAMETER,
    GAUSPULS_ERROR_ARG_COUNT,
    GAUSPULS_ERROR_INTERNAL,
];

pub const GAUSPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GAUSPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GAUSPULS_ERRORS,
};

const GAUSPULS_INTEGER_TIME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-integer-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with typed-integer sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsIntegerTimeExtension"),
};

const GAUSPULS_LOGICAL_TIME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-logical-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with logical sample times is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsLogicalTimeExtension"),
};

const GAUSPULS_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-integer-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with a typed-integer scalar control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsIntegerControlExtension"),
};

const GAUSPULS_LOGICAL_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-logical-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with a logical scalar control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsLogicalControlExtension"),
};

const GAUSPULS_SINGLE_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-single-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with a single-precision scalar control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsSingleControlExtension"),
};

const GAUSPULS_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gauspuls-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gauspuls with an interactive resident input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GauspulsResidentInputExtension"),
};

pub const GAUSPULS_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    GAUSPULS_INTEGER_TIME_EXTENSION,
    GAUSPULS_LOGICAL_TIME_EXTENSION,
    GAUSPULS_INTEGER_CONTROL_EXTENSION,
    GAUSPULS_LOGICAL_CONTROL_EXTENSION,
    GAUSPULS_SINGLE_CONTROL_EXTENSION,
    GAUSPULS_RESIDENT_INPUT_EXTENSION,
];

const GAUSPULS_INTEGER_TIME_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "T",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are read from authoritative storage and admitted only when exactly representable as binary64 sample times.",
    }];

const GAUSPULS_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "FC, BW, BWR, or TPE",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are scalar-only and must be exactly representable at the binary64 signal-computation boundary.",
    }];

pub const GAUSPULS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = gauspuls(integer_T, FC, BW, BWR)",
        inputs: &GAUSPULS_INTEGER_TIME_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only integer sample times cross one exact binary64 boundary and produce host double output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y or TC = gauspuls(..., integer_control, ...)",
        inputs: &GAUSPULS_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only scalar controls cross one exact binary64 boundary; output follows the documented floating form.",
    },
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct GauspulsParams {
    pub fc: f64,
    pub bw: f64,
    pub bwr: f64,
}

fn gauspuls_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gauspuls_error_with_message(error.message, error)
}

fn gauspuls_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    gauspuls_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn gauspuls_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn gauspuls_error_with_source(
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

pub(crate) fn default_params() -> GauspulsParams {
    GauspulsParams {
        fc: DEFAULT_FC,
        bw: DEFAULT_BW,
        bwr: DEFAULT_BWR,
    }
}

pub(crate) fn validate_params(params: GauspulsParams) -> Result<GauspulsParams, String> {
    if !params.fc.is_finite() || params.fc <= 0.0 {
        return Err(format!(
            "carrier frequency must be positive and finite, got {}",
            params.fc
        ));
    }
    if !params.bw.is_finite() || params.bw <= 0.0 {
        return Err(format!(
            "bandwidth must be positive and finite, got {}",
            params.bw
        ));
    }
    if !params.bwr.is_finite() || params.bwr >= 0.0 {
        return Err(format!(
            "bandwidth reference must be negative and finite, got {}",
            params.bwr
        ));
    }
    Ok(params)
}

pub(crate) fn validate_tpe(tpe: f64) -> Result<f64, String> {
    if !tpe.is_finite() || tpe >= 0.0 {
        Err(format!(
            "cutoff envelope level must be negative and finite, got {tpe}"
        ))
    } else {
        Ok(tpe)
    }
}

pub(crate) fn gauspuls_scalar(t: f64, params: GauspulsParams) -> f64 {
    let (in_phase, _, _) = gauspuls_components_scalar(t, params);
    in_phase
}

pub(crate) fn gauspuls_components_scalar(t: f64, params: GauspulsParams) -> (f64, f64, f64) {
    if t.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if !t.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let a = gaussian_shape_factor(params);
    let envelope = (-a * t * t).exp();
    let phase = 2.0 * PI * params.fc * t;
    (envelope * phase.cos(), envelope * phase.sin(), envelope)
}

pub(crate) fn gauspuls_cutoff(params: GauspulsParams, tpe: f64) -> f64 {
    let a = gaussian_shape_factor(params);
    (-db_to_log_amplitude(tpe) / a).sqrt()
}

pub(crate) fn gauspuls_tensor(tensor: Tensor, params: GauspulsParams) -> Result<Tensor, String> {
    let shape = tensor.shape.clone();
    ensure_exact_integer_tensor_boundary(&tensor, "sample time")?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| err.to_string())?;
    let storage = match storage {
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| gauspuls_scalar(f64::from(value), params) as f32)
                .collect(),
        ),
        storage => NumericStorage::F64(
            storage
                .materialize_f64()
                .into_iter()
                .map(|value| gauspuls_scalar(value, params))
                .collect(),
        ),
    };
    Tensor::from_numeric_storage(storage, shape).map_err(|err| err.to_string())
}

pub(crate) fn gauspuls_components_tensor(
    tensor: Tensor,
    params: GauspulsParams,
) -> Result<(Tensor, Tensor, Tensor), String> {
    let shape = tensor.shape.clone();
    ensure_exact_integer_tensor_boundary(&tensor, "sample time")?;
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| err.to_string())?;
    let output_is_single = matches!(&storage, NumericStorage::F32(_));
    let values = storage.materialize_f64();
    let mut in_phase = Vec::with_capacity(values.len());
    let mut quadrature = Vec::with_capacity(values.len());
    let mut envelope = Vec::with_capacity(values.len());
    for value in values {
        let (yi, yq, ye) = gauspuls_components_scalar(value, params);
        in_phase.push(yi);
        quadrature.push(yq);
        envelope.push(ye);
    }
    let build = |values: Vec<f64>, shape: Vec<usize>| {
        let storage = if output_is_single {
            NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
        } else {
            NumericStorage::F64(values)
        };
        Tensor::from_numeric_storage(storage, shape).map_err(|err| err.to_string())
    };
    Ok((
        build(in_phase, shape.clone())?,
        build(quadrature, shape.clone())?,
        build(envelope, shape)?,
    ))
}

fn ensure_exact_integer_tensor_boundary(tensor: &Tensor, role: &str) -> Result<(), String> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    if storage
        .exact_values()
        .iter()
        .all(crate::builtins::math::trigonometry::cos::integer_is_exact_f64)
    {
        Ok(())
    } else {
        Err(format!(
            "integer {role} must be exactly representable as double"
        ))
    }
}

fn gaussian_shape_factor(params: GauspulsParams) -> f64 {
    let numerator = -(PI * params.fc * params.bw).powi(2);
    numerator / (4.0 * db_to_log_amplitude(params.bwr))
}

fn db_to_log_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0).ln()
}

#[runtime_builtin(
    name = "gauspuls",
    category = "math/signal",
    summary = "Generate Gaussian-modulated sinusoidal pulses.",
    keywords = "gauspuls,gaussian pulse,pulse train,signal processing,cutoff",
    type_resolver(numeric_unary_shape_type),
    descriptor(crate::builtins::math::signal::gauspuls::GAUSPULS_DESCRIPTOR),
    extensions(crate::builtins::math::signal::gauspuls::GAUSPULS_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::gauspuls::GAUSPULS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::gauspuls"
)]
async fn gauspuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(mode) = text_scalar(&t) {
        return gauspuls_mode(mode, rest).await;
    }
    if rest.len() > 3 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    ensure_gauspuls_time_extensions(&t)?;
    ensure_gauspuls_control_extensions(&rest)?;
    let params = parse_params(&rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        return gauspuls_with_output_count(t, params, out_count).await;
    }
    gauspuls_value(t, params).await
}

fn ensure_gauspuls_time_extensions(value: &Value) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_INTEGER_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_LOGICAL_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAUSPULS_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn ensure_gauspuls_control_extensions(values: &[Value]) -> BuiltinResult<()> {
    for value in values {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_INTEGER_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_LOGICAL_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_single_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_SINGLE_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::GpuTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GAUSPULS_RESIDENT_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

async fn gauspuls_value(t: Value, params: GauspulsParams) -> BuiltinResult<Value> {
    match t {
        Value::GpuTensor(handle) => gauspuls_gpu(handle, params).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(gauspuls_error(&GAUSPULS_ERROR_INVALID_INPUT))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(gauspuls_error(&GAUSPULS_ERROR_INVALID_INPUT))
        }
        other => gauspuls_real(other, params),
    }
}

async fn gauspuls_with_output_count(
    t: Value,
    params: GauspulsParams,
    out_count: usize,
) -> BuiltinResult<Value> {
    if out_count == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }
    if out_count <= 1 {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![gauspuls_value(t, params).await?],
        ));
    }
    let (in_phase, quadrature, envelope) = gauspuls_components_value(t, params).await?;
    Ok(crate::output_count::output_list_with_padding(
        out_count,
        vec![in_phase, quadrature, envelope],
    ))
}

async fn gauspuls_mode(mode: String, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !mode.eq_ignore_ascii_case("cutoff") {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_INVALID_INPUT,
            format!("unsupported mode '{mode}'"),
        ));
    }
    if rest.len() > 4 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    ensure_gauspuls_control_extensions(&rest)?;
    let params = parse_params(&rest[..rest.len().min(3)]).await?;
    let tpe = match rest.get(3) {
        Some(value) if !is_empty_default_placeholder(value) => {
            validate_tpe(scalar_parameter(value, "TPE").await?).map_err(|err| {
                gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str())
            })?
        }
        _ => DEFAULT_TPE,
    };
    let value = Value::Num(gauspuls_cutoff(params, tpe));
    if let Some(out_count) = crate::output_count::current_output_count() {
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![value],
        ));
    }
    Ok(value)
}

async fn parse_params(rest: &[Value]) -> BuiltinResult<GauspulsParams> {
    if rest.len() > 3 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    let mut params = default_params();
    if let Some(value) = rest
        .first()
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.fc = scalar_parameter(value, "FC").await?;
    }
    if let Some(value) = rest
        .get(1)
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.bw = scalar_parameter(value, "BW").await?;
    }
    if let Some(value) = rest
        .get(2)
        .filter(|value| !is_empty_default_placeholder(value))
    {
        params.bwr = scalar_parameter(value, "BWR").await?;
    }
    validate_params(params)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str()))
}

fn is_empty_default_placeholder(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.len() == 0)
}

async fn scalar_parameter(value: &Value, label: &str) -> BuiltinResult<f64> {
    let host = crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(|err| {
            gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?;
    if let Some(integer) = tensor::scalar_integer_value(&host) {
        if !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&integer) {
            return Err(gauspuls_error_with_detail(
                &GAUSPULS_ERROR_INVALID_PARAMETER,
                format!("{label}: integer value must be exactly representable as double"),
            ));
        }
    }
    scalar_f64_from_value_async(&host)
        .await
        .map_err(|err| {
            gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, format!("{label}: {err}"))
        })?
        .ok_or_else(|| {
            gauspuls_error_with_detail(
                &GAUSPULS_ERROR_INVALID_PARAMETER,
                format!("{label}: expected scalar"),
            )
        })
}

async fn gauspuls_gpu(handle: GpuTensorHandle, params: GauspulsParams) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|source| {
            gauspuls_error_with_source(&GAUSPULS_ERROR_INTERNAL, "gpu gather failed", source)
        })?;
    gauspuls_tensor(tensor, params)
        .map(tensor_into_value)
        .map_err(map_gauspuls_tensor_error)
}

fn gauspuls_real(value: Value, params: GauspulsParams) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_INPUT, err))?;
    gauspuls_tensor(tensor, params)
        .map(tensor_into_value)
        .map_err(map_gauspuls_tensor_error)
}

async fn gauspuls_components_value(
    value: Value,
    params: GauspulsParams,
) -> BuiltinResult<(Value, Value, Value)> {
    let tensor = match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    gauspuls_error_with_source(
                        &GAUSPULS_ERROR_INTERNAL,
                        "gpu gather failed",
                        source,
                    )
                })?
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(gauspuls_error(&GAUSPULS_ERROR_INVALID_INPUT));
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err(gauspuls_error(&GAUSPULS_ERROR_INVALID_INPUT));
        }
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_INPUT, err))?,
    };
    let (in_phase, quadrature, envelope) =
        gauspuls_components_tensor(tensor, params).map_err(map_gauspuls_tensor_error)?;
    Ok((
        tensor_into_value(in_phase),
        tensor_into_value(quadrature),
        tensor_into_value(envelope),
    ))
}

fn map_gauspuls_tensor_error(error: String) -> RuntimeError {
    let descriptor = if error.contains("exactly representable as double") {
        &GAUSPULS_ERROR_INVALID_INPUT
    } else {
        &GAUSPULS_ERROR_INTERNAL
    };
    gauspuls_error_with_detail(descriptor, error)
}

fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        builtin_function_by_name, CharArray, IntValue, IntegerStorage, NumericStorage,
    };

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(gauspuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor");
        tensor
    }

    fn all_integer_time_tensors() -> Vec<Tensor> {
        vec![
            Tensor::new_integer(IntegerStorage::I8(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I32(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I64(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U32(vec![0]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1]).unwrap(),
        ]
    }

    fn all_integer_controls() -> Vec<Value> {
        vec![
            Value::Int(IntValue::I8(1)),
            Value::Int(IntValue::I16(1)),
            Value::Int(IntValue::I32(1)),
            Value::Int(IntValue::I64(1)),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U16(1)),
            Value::Int(IntValue::U32(1)),
            Value::Int(IntValue::U64(1)),
        ]
    }

    #[test]
    fn gauspuls_defaults_peak_at_one() {
        let out = call(Value::Num(0.0), Vec::new()).expect("gauspuls");
        assert!(matches!(out, Value::Num(value) if (value - 1.0).abs() <= 1e-12));
    }

    #[test]
    fn gauspuls_custom_parameters_preserve_shape() {
        let input = Tensor::new(vec![-0.001, 0.0, 0.001], vec![1, 3]).unwrap();
        let out = expect_tensor(
            call(
                Value::Tensor(input),
                vec![Value::Num(1000.0), Value::Num(0.5), Value::Num(-6.0)],
            )
            .expect("gauspuls"),
        );
        assert_eq!(out.shape, vec![1, 3]);
        assert!(out.materialize_f64()[1] > out.materialize_f64()[0]);
        assert!(out.materialize_f64()[1] > out.materialize_f64()[2]);
    }

    #[test]
    fn gauspuls_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = integer_tensor(vec![0, 1], vec![1, 2]);
        let out = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("gauspuls"));
        assert_eq!(out.shape, vec![1, 2]);
        assert!((out.materialize_f64()[0] - 1.0).abs() <= 1e-12);
        assert!(out.materialize_f64()[1].is_finite());
    }

    #[test]
    fn gauspuls_reads_all_integer_time_classes_from_authoritative_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for input in all_integer_time_tensors() {
            let output = call(Value::Tensor(input), Vec::new()).expect("gauspuls");
            assert!(matches!(output, Value::Num(value) if value == 1.0));
        }
    }

    #[test]
    fn gauspuls_accepts_all_integer_scalar_control_classes() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for control in all_integer_controls() {
            let output = call(Value::Num(0.0), vec![control]).expect("integer FC");
            assert!(matches!(output, Value::Num(value) if value == 1.0));
        }
    }

    #[test]
    fn gauspuls_rejects_wide_integers_at_binary64_boundaries() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = (1_u64 << 53) + 1;
        let input = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let error = call(Value::Tensor(input), Vec::new()).expect_err("wide sample time");
        assert_eq!(error.identifier(), GAUSPULS_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("exactly representable as double"));

        let error = call(Value::Num(0.0), vec![Value::Int(IntValue::U64(wide))])
            .expect_err("wide scalar control");
        assert_eq!(
            error.identifier(),
            GAUSPULS_ERROR_INVALID_PARAMETER.identifier
        );
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn gauspuls_preserves_native_single_time_shape_and_output_class() {
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![-0.001, 0.0, 0.001]), vec![1, 3])
                .unwrap();
        let output = expect_tensor(call(Value::Tensor(input), Vec::new()).expect("single T"));
        assert_eq!(output.shape, vec![1, 3]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);

        let _count = crate::output_count::push_output_count(Some(3));
        let input =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 0.00025]), vec![2, 1])
                .unwrap();
        let Value::OutputList(outputs) = call(Value::Tensor(input), Vec::new()).expect("outputs")
        else {
            panic!("expected output list");
        };
        for output in outputs {
            let output = expect_tensor(output);
            assert_eq!(output.shape, vec![2, 1]);
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
        }
    }

    #[test]
    fn gauspuls_extension_roles_and_compatibility_order_are_stable() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let integer_time =
            call(Value::Int(IntValue::U8(0)), Vec::new()).expect_err("integer time extension");
        assert_eq!(
            integer_time.identifier(),
            GAUSPULS_INTEGER_TIME_EXTENSION.error_identifier
        );

        let logical_time =
            call(Value::Bool(false), Vec::new()).expect_err("logical time extension");
        assert_eq!(
            logical_time.identifier(),
            GAUSPULS_LOGICAL_TIME_EXTENSION.error_identifier
        );

        let integer_control = call(Value::Num(0.0), vec![Value::Int(IntValue::U8(1))])
            .expect_err("integer control extension");
        assert_eq!(
            integer_control.identifier(),
            GAUSPULS_INTEGER_CONTROL_EXTENSION.error_identifier
        );

        let logical_control =
            call(Value::Num(0.0), vec![Value::Bool(true)]).expect_err("logical control extension");
        assert_eq!(
            logical_control.identifier(),
            GAUSPULS_LOGICAL_CONTROL_EXTENSION.error_identifier
        );

        let single_control =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0]), vec![1, 1]).unwrap();
        let single_control = call(Value::Num(0.0), vec![Value::Tensor(single_control)])
            .expect_err("single control extension");
        assert_eq!(
            single_control.identifier(),
            GAUSPULS_SINGLE_CONTROL_EXTENSION.error_identifier
        );

        let mut resident = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX - 8,
            buffer_id: u64::MAX - 8,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let resident_error = call(Value::GpuTensor(resident.clone()), Vec::new())
            .expect_err("resident extension before provider access");
        assert_eq!(
            resident_error.identifier(),
            GAUSPULS_RESIDENT_INPUT_EXTENSION.error_identifier
        );
        let resident_control_error =
            call(Value::Num(0.0), vec![Value::GpuTensor(resident.clone())])
                .expect_err("resident control extension before provider access");
        assert_eq!(
            resident_control_error.identifier(),
            GAUSPULS_RESIDENT_INPUT_EXTENSION.error_identifier
        );

        resident.descriptor.element_type = Some(runmat_accelerate_api::NumericElementType::U64);
        let typed_resident_error = call(Value::GpuTensor(resident.clone()), Vec::new())
            .expect_err("integer role precedes residency");
        assert_eq!(
            typed_resident_error.identifier(),
            GAUSPULS_INTEGER_TIME_EXTENSION.error_identifier
        );
    }

    #[test]
    fn gauspuls_integer_capabilities_cover_time_and_scalar_controls() {
        assert_eq!(GAUSPULS_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(GAUSPULS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(GAUSPULS_INTEGER_CAPABILITIES[1].inputs[0].classes.len(), 8);
        assert_eq!(
            GAUSPULS_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Double
        );
    }

    #[test]
    fn gauspuls_multi_output_returns_quadrature_and_envelope() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let input = Tensor::new(vec![0.0, 0.00025], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(input),
            vec![Value::Num(1000.0), Value::Num(0.5), Value::Num(-6.0)],
        )
        .expect("gauspuls");
        let Value::OutputList(outputs) = out else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 3);
        let in_phase = expect_tensor(outputs[0].clone());
        let quadrature = expect_tensor(outputs[1].clone());
        let envelope = expect_tensor(outputs[2].clone());
        assert_eq!(in_phase.shape, vec![1, 2]);
        assert_eq!(quadrature.shape, vec![1, 2]);
        assert_eq!(envelope.shape, vec![1, 2]);
        assert!((in_phase.materialize_f64()[0] - 1.0).abs() <= 1e-12);
        assert!(quadrature.materialize_f64()[0].abs() <= 1e-12);
        assert!((envelope.materialize_f64()[0] - 1.0).abs() <= 1e-12);
        assert!(in_phase.materialize_f64()[1].abs() <= 1e-12);
        assert!((quadrature.materialize_f64()[1] - envelope.materialize_f64()[1]).abs() <= 1e-12);
    }

    #[test]
    fn gauspuls_multi_output_reads_typed_integer_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = crate::output_count::push_output_count(Some(3));
        let input = integer_tensor(vec![0, 1], vec![1, 2]);
        let out = call(Value::Tensor(input), Vec::new()).expect("gauspuls");
        let Value::OutputList(outputs) = out else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 3);
        let in_phase = expect_tensor(outputs[0].clone());
        let quadrature = expect_tensor(outputs[1].clone());
        let envelope = expect_tensor(outputs[2].clone());
        assert_eq!(in_phase.shape, vec![1, 2]);
        assert_eq!(quadrature.shape, vec![1, 2]);
        assert_eq!(envelope.shape, vec![1, 2]);
        assert!((in_phase.materialize_f64()[0] - 1.0).abs() <= 1e-12);
        assert!(quadrature.materialize_f64()[0].abs() <= 1e-12);
        assert!((envelope.materialize_f64()[0] - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn gauspuls_cutoff_mode_returns_positive_time() {
        let out = call(
            Value::CharArray(CharArray::new_row("cutoff")),
            vec![
                Value::Num(1000.0),
                Value::Num(0.5),
                Value::Num(-6.0),
                Value::Num(-60.0),
            ],
        )
        .expect("cutoff");
        assert!(matches!(out, Value::Num(value) if value > 0.0 && value < 0.01));
    }

    #[test]
    fn gauspuls_cutoff_accepts_documented_empty_default_placeholder() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty double placeholder");
        let output = call(
            Value::CharArray(CharArray::new_row("cutoff")),
            vec![
                Value::Num(50_000.0),
                Value::Num(0.6),
                Value::Tensor(empty),
                Value::Num(-40.0),
            ],
        )
        .expect("documented cutoff form");
        assert!(matches!(output, Value::Num(value) if value.is_finite() && value > 0.0));
    }

    #[test]
    fn gauspuls_cutoff_mode_honors_requested_output_count() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = call(
            Value::CharArray(CharArray::new_row("cutoff")),
            vec![
                Value::Num(1000.0),
                Value::Num(0.5),
                Value::Num(-6.0),
                Value::Num(-60.0),
            ],
        )
        .expect("cutoff");
        let Value::OutputList(outputs) = out else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        assert!(matches!(outputs[0], Value::Num(value) if value > 0.0 && value < 0.01));
        assert!(matches!(outputs[1], Value::Num(value) if value == 0.0));
    }

    #[test]
    fn gauspuls_rejects_bad_parameters() {
        let err = call(Value::Num(0.0), vec![Value::Num(0.0)]).expect_err("fc");
        assert_eq!(
            err.identifier(),
            GAUSPULS_ERROR_INVALID_PARAMETER.identifier
        );

        let err = call(Value::Num(0.0), vec![Value::Num(1000.0), Value::Num(0.0)]).expect_err("bw");
        assert_eq!(
            err.identifier(),
            GAUSPULS_ERROR_INVALID_PARAMETER.identifier
        );
    }

    #[test]
    fn gauspuls_is_registered() {
        assert!(builtin_function_by_name("gauspuls").is_some());
    }
}
