//! MATLAB-compatible `buttord` Butterworth order selection.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    parse_scalar_f64, real_vector_to_row_value, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::buttord_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "buttord";
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::buttord")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "buttord",
    op_kind: GpuOpKind::Custom("butterworth-order-selection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Butterworth order selection is scalar host-side analysis; GPU inputs are gathered automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::buttord")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "buttord",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "buttord materialises scalar/vector design parameters and is not fused.",
};

const BUTTORD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum Butterworth filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Natural cutoff frequency or frequency pair for butter.",
    },
];

const BUTTORD_INPUTS_DIGITAL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Passband edge or two-element passband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stopband edge or two-element stopband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
];

const BUTTORD_INPUTS_ANALOG: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog passband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog stopband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog-design selector.",
    },
];

const BUTTORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs)",
        inputs: &BUTTORD_INPUTS_DIGITAL,
        outputs: &BUTTORD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs, 's')",
        inputs: &BUTTORD_INPUTS_ANALOG,
        outputs: &BUTTORD_OUTPUTS,
    },
];

const BUTTORD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.ARG_COUNT",
    identifier: Some("RunMat:buttord:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "buttord: expected buttord(Wp, Ws, Rp, Rs [, 's'])",
};

const BUTTORD_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_FREQUENCY",
    identifier: Some("RunMat:buttord:InvalidFrequency"),
    when: "Passband or stopband edges are invalid.",
    message: "buttord: invalid passband or stopband frequency",
};

const BUTTORD_ERROR_INVALID_ATTENUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_ATTENUATION",
    identifier: Some("RunMat:buttord:InvalidAttenuation"),
    when: "Ripple or attenuation values are invalid.",
    message: "buttord: Rp and Rs must be positive finite scalars with Rs > Rp",
};

const BUTTORD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_OPTION",
    identifier: Some("RunMat:buttord:InvalidOption"),
    when: "The optional selector is not supported.",
    message: "buttord: optional selector must be 's'",
};

const BUTTORD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INTERNAL",
    identifier: Some("RunMat:buttord:Internal"),
    when: "Output tensor construction fails internally.",
    message: "buttord: internal error",
};

const BUTTORD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    BUTTORD_ERROR_ARG_COUNT,
    BUTTORD_ERROR_INVALID_FREQUENCY,
    BUTTORD_ERROR_INVALID_ATTENUATION,
    BUTTORD_ERROR_INVALID_OPTION,
    BUTTORD_ERROR_INTERNAL,
];

pub const BUTTORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUTTORD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUTTORD_ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Clone, Debug)]
struct ButtordResult {
    order: usize,
    wn: Vec<f64>,
}

fn buttord_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    buttord_error_with_message(error.message, error)
}

fn buttord_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    buttord_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn buttord_error_with_message(
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
    name = "buttord",
    category = "math/signal",
    summary = "Select the minimum Butterworth filter order.",
    keywords = "buttord,butterworth,filter order,signal processing",
    type_resolver(buttord_type),
    descriptor(crate::builtins::math::signal::buttord::BUTTORD_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::buttord"
)]
async fn buttord_builtin(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    evaluate(wp, ws, rp, rs, &rest).await
}

pub async fn evaluate(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: &[Value],
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(buttord_error(&BUTTORD_ERROR_ARG_COUNT));
    }
    let analog = if let Some(option) = rest.first() {
        match option_text(option).as_deref() {
            Some("s") => true,
            Some(_) => return Err(buttord_error(&BUTTORD_ERROR_INVALID_OPTION)),
            None => return Err(buttord_error(&BUTTORD_ERROR_INVALID_OPTION)),
        }
    } else {
        false
    };
    let wp = real_edges("Wp", wp).await?;
    let ws = real_edges("Ws", ws).await?;
    let rp = parse_positive_db("Rp", &rp)?;
    let rs = parse_positive_db("Rs", &rs)?;
    if rs <= rp {
        return Err(buttord_error(&BUTTORD_ERROR_INVALID_ATTENUATION));
    }

    let result = compute_buttord(&wp, &ws, rp, rs, analog)?;
    output_result(result)
}

fn output_result(result: ButtordResult) -> BuiltinResult<Value> {
    let order = Value::Num(result.order as f64);
    let wn = if result.wn.len() == 1 {
        Value::Num(result.wn[0])
    } else {
        real_vector_to_row_value(result.wn)
            .map_err(|err| buttord_error_with_detail(&BUTTORD_ERROR_INTERNAL, err.message()))?
    };
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![order]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![order, wn],
        ));
    }
    Ok(order)
}

async fn real_edges(label: &'static str, value: Value) -> BuiltinResult<Vec<f64>> {
    let input = value_to_complex_vector(BUILTIN_NAME, label, value)
        .await
        .map_err(|err| {
            buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, err.message())
        })?;
    if input.data.len() != 1 && input.data.len() != 2 {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            format!("{label} must be a scalar or two-element vector"),
        ));
    }
    let mut out = Vec::with_capacity(input.data.len());
    for value in input.data {
        if value.im.abs() > EPS || !value.re.is_finite() || value.re <= 0.0 {
            return Err(buttord_error_with_detail(
                &BUTTORD_ERROR_INVALID_FREQUENCY,
                format!("{label} entries must be positive finite real values"),
            ));
        }
        out.push(value.re);
    }
    if out.len() == 2 && out[0] >= out[1] {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            format!("{label} edge vector must be strictly increasing"),
        ));
    }
    Ok(out)
}

fn parse_positive_db(label: &'static str, value: &Value) -> BuiltinResult<f64> {
    let parsed = parse_scalar_f64(BUILTIN_NAME, label, value).map_err(|err| {
        buttord_error_with_detail(&BUTTORD_ERROR_INVALID_ATTENUATION, err.message())
    })?;
    if parsed <= 0.0 {
        return Err(buttord_error(&BUTTORD_ERROR_INVALID_ATTENUATION));
    }
    Ok(parsed)
}

fn option_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.trim().to_ascii_lowercase()),
        Value::StringArray(array) if array.data.len() == 1 => {
            Some(array.data[0].trim().to_ascii_lowercase())
        }
        Value::CharArray(chars) if chars.rows <= 1 => Some(
            chars
                .data
                .iter()
                .collect::<String>()
                .trim()
                .to_ascii_lowercase(),
        ),
        _ => None,
    }
}

fn compute_buttord(
    wp: &[f64],
    ws: &[f64],
    rp: f64,
    rs: f64,
    analog: bool,
) -> BuiltinResult<ButtordResult> {
    if wp.len() != ws.len() {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            "Wp and Ws must have the same length",
        ));
    }
    let kind = classify_kind(wp, ws)?;
    validate_domain(wp, analog, "Wp")?;
    validate_domain(ws, analog, "Ws")?;

    let wp_work = transform_edges(wp, analog);
    let ws_work = transform_edges(ws, analog);
    let epsilon = (10.0f64.powf(rp / 10.0) - 1.0).sqrt();
    let stop = 10.0f64.powf(rs / 10.0) - 1.0;
    let nat = stopband_ratio(kind, &wp_work, &ws_work)?;
    if nat <= 1.0 || !nat.is_finite() {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            "stopband must be separated from passband",
        ));
    }
    let order = ((stop / (epsilon * epsilon)).log10() / (2.0 * nat.log10())).ceil() as usize;
    let order = order.max(1);
    let natural = natural_frequency(kind, &wp_work, &ws_work, epsilon, order)?;
    let wn = if analog {
        natural
    } else {
        natural.into_iter().map(unwarp_frequency).collect()
    };
    Ok(ButtordResult { order, wn })
}

fn classify_kind(wp: &[f64], ws: &[f64]) -> BuiltinResult<FilterKind> {
    match (wp.len(), ws.len()) {
        (1, 1) if wp[0] < ws[0] => Ok(FilterKind::Lowpass),
        (1, 1) if wp[0] > ws[0] => Ok(FilterKind::Highpass),
        (2, 2) if ws[0] < wp[0] && wp[1] < ws[1] => Ok(FilterKind::Bandpass),
        (2, 2) if wp[0] < ws[0] && ws[1] < wp[1] => Ok(FilterKind::Bandstop),
        _ => Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            "passband and stopband edges do not define a supported Butterworth response",
        )),
    }
}

fn validate_domain(edges: &[f64], analog: bool, label: &'static str) -> BuiltinResult<()> {
    if !analog && edges.iter().any(|&edge| edge >= 1.0) {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            format!("{label} digital frequencies must be between 0 and 1"),
        ));
    }
    Ok(())
}

fn transform_edges(edges: &[f64], analog: bool) -> Vec<f64> {
    if analog {
        edges.to_vec()
    } else {
        edges.iter().map(|&edge| prewarp_frequency(edge)).collect()
    }
}

fn prewarp_frequency(edge: f64) -> f64 {
    (std::f64::consts::PI * edge / 2.0).tan()
}

fn unwarp_frequency(edge: f64) -> f64 {
    2.0 * edge.atan() / std::f64::consts::PI
}

fn stopband_ratio(kind: FilterKind, wp: &[f64], ws: &[f64]) -> BuiltinResult<f64> {
    match kind {
        FilterKind::Lowpass => Ok(ws[0] / wp[0]),
        FilterKind::Highpass => Ok(wp[0] / ws[0]),
        FilterKind::Bandpass => {
            let bandwidth = wp[1] - wp[0];
            let center_sq = wp[0] * wp[1];
            Ok(ws
                .iter()
                .map(|&edge| ((edge * edge - center_sq) / (bandwidth * edge)).abs())
                .fold(f64::INFINITY, f64::min))
        }
        FilterKind::Bandstop => {
            let bandwidth = ws[1] - ws[0];
            let center_sq = ws[0] * ws[1];
            Ok(wp
                .iter()
                .map(|&edge| (bandwidth * edge / (edge * edge - center_sq)).abs())
                .fold(f64::INFINITY, f64::min))
        }
    }
}

fn natural_frequency(
    kind: FilterKind,
    wp: &[f64],
    ws: &[f64],
    epsilon: f64,
    order: usize,
) -> BuiltinResult<Vec<f64>> {
    let cutoff_scale = epsilon.powf(-1.0 / order as f64);
    match kind {
        FilterKind::Lowpass => Ok(vec![wp[0] * cutoff_scale]),
        FilterKind::Highpass => Ok(vec![wp[0] / cutoff_scale]),
        FilterKind::Bandpass => {
            let bandwidth = wp[1] - wp[0];
            let center_sq = wp[0] * wp[1];
            let disc = (cutoff_scale * bandwidth).powi(2) + 4.0 * center_sq;
            let root = disc.sqrt();
            Ok(vec![
                (-cutoff_scale * bandwidth + root) / 2.0,
                (cutoff_scale * bandwidth + root) / 2.0,
            ])
        }
        FilterKind::Bandstop => {
            let bandwidth = ws[1] - ws[0];
            let center_sq = ws[0] * ws[1];
            let disc = bandwidth * bandwidth + 4.0 * cutoff_scale * cutoff_scale * center_sq;
            let root = disc.sqrt();
            Ok(vec![
                (-bandwidth + root) / (2.0 * cutoff_scale),
                (bandwidth + root) / (2.0 * cutoff_scale),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, Tensor};

    fn call(
        wp: Value,
        ws: Value,
        rp: Value,
        rs: Value,
        rest: &[Value],
        outputs: Option<usize>,
    ) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(wp, ws, rp, rs, rest))
    }

    fn outputs(value: Value) -> (f64, Vec<f64>) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        let order = match &values[0] {
            Value::Num(value) => *value,
            other => panic!("expected scalar order, got {other:?}"),
        };
        let wn = match &values[1] {
            Value::Num(value) => vec![*value],
            Value::Tensor(tensor) => tensor.data.clone(),
            other => panic!("expected Wn, got {other:?}"),
        };
        (order, wn)
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("buttord builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[n, Wn] = buttord(Wp, Ws, Rp, Rs)"));
    }

    #[test]
    fn digital_lowpass_order_and_cutoff() {
        let out = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 12.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.2108).abs() < 5e-4);
    }

    #[test]
    fn digital_highpass_order_and_cutoff() {
        let out = call(
            Value::Num(0.5),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 8.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.4732).abs() < 5e-4);
    }

    #[test]
    fn analog_lowpass_keeps_radian_cutoff() {
        let out = call(
            Value::Num(10.0),
            Value::Num(20.0),
            Value::Num(1.0),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 8.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 12.0);
    }

    #[test]
    fn bandpass_returns_two_cutoff_edges() {
        let wp = Tensor::new(vec![0.3, 0.5], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.2, 0.6], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(1.0),
            Value::Num(30.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert!(n >= 1.0);
        assert_eq!(wn.len(), 2);
        assert!(wn[0] < 0.3);
        assert!(wn[1] > 0.5);
        assert!(wn[0] < wn[1]);
    }

    #[test]
    fn rejects_invalid_specs() {
        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(40.0),
            Value::Num(1.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:InvalidAttenuation"));

        let err = call(
            Value::Num(1.2),
            Value::Num(1.4),
            Value::Num(1.0),
            Value::Num(20.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:InvalidFrequency"));
    }
}
