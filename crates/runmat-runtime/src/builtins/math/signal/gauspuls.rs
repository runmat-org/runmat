//! MATLAB-compatible `gauspuls` builtin for Gaussian-modulated sinusoid samples.

use std::f64::consts::PI;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
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
    let data = tensor
        .data
        .iter()
        .map(|&value| gauspuls_scalar(value, params))
        .collect::<Vec<_>>();
    Tensor::new(data, shape).map_err(|err| err.to_string())
}

pub(crate) fn gauspuls_components_tensor(
    tensor: Tensor,
    params: GauspulsParams,
) -> Result<(Tensor, Tensor, Tensor), String> {
    let shape = tensor.shape.clone();
    let mut in_phase = Vec::with_capacity(tensor.data.len());
    let mut quadrature = Vec::with_capacity(tensor.data.len());
    let mut envelope = Vec::with_capacity(tensor.data.len());
    for value in tensor.data {
        let (yi, yq, ye) = gauspuls_components_scalar(value, params);
        in_phase.push(yi);
        quadrature.push(yq);
        envelope.push(ye);
    }
    Ok((
        Tensor::new(in_phase, shape.clone()).map_err(|err| err.to_string())?,
        Tensor::new(quadrature, shape.clone()).map_err(|err| err.to_string())?,
        Tensor::new(envelope, shape).map_err(|err| err.to_string())?,
    ))
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
    builtin_path = "crate::builtins::math::signal::gauspuls"
)]
async fn gauspuls_builtin(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if let Some(mode) = text_scalar(&t) {
        return gauspuls_mode(mode, rest).await;
    }
    let params = parse_params(&rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        return gauspuls_with_output_count(t, params, out_count).await;
    }
    gauspuls_value(t, params).await
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
    let params = parse_params(&rest[..rest.len().min(3)]).await?;
    let tpe = match rest.get(3) {
        Some(value) => validate_tpe(scalar_parameter(value, "TPE").await?).map_err(|err| {
            gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str())
        })?,
        None => DEFAULT_TPE,
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
    if let Some(value) = rest.first() {
        params.fc = scalar_parameter(value, "FC").await?;
    }
    if let Some(value) = rest.get(1) {
        params.bw = scalar_parameter(value, "BW").await?;
    }
    if let Some(value) = rest.get(2) {
        params.bwr = scalar_parameter(value, "BWR").await?;
    }
    validate_params(params)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_PARAMETER, err.as_str()))
}

async fn scalar_parameter(value: &Value, label: &str) -> BuiltinResult<f64> {
    scalar_f64_from_value_async(value)
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
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INTERNAL, err))
}

fn gauspuls_real(value: Value, params: GauspulsParams) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INVALID_INPUT, err))?;
    gauspuls_tensor(tensor, params)
        .map(tensor_into_value)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INTERNAL, err))
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
    let (in_phase, quadrature, envelope) = gauspuls_components_tensor(tensor, params)
        .map_err(|err| gauspuls_error_with_detail(&GAUSPULS_ERROR_INTERNAL, err))?;
    Ok((
        tensor_into_value(in_phase),
        tensor_into_value(quadrature),
        tensor_into_value(envelope),
    ))
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
    use runmat_builtins::{builtin_function_by_name, CharArray};

    fn call(t: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(gauspuls_builtin(t, rest))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
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
        assert!(out.data[1] > out.data[0]);
        assert!(out.data[1] > out.data[2]);
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
        assert!((in_phase.data[0] - 1.0).abs() <= 1e-12);
        assert!(quadrature.data[0].abs() <= 1e-12);
        assert!((envelope.data[0] - 1.0).abs() <= 1e-12);
        assert!(in_phase.data[1].abs() <= 1e-12);
        assert!((quadrature.data[1] - envelope.data[1]).abs() <= 1e-12);
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
