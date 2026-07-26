//! MATLAB-compatible `cheb2ord` Chebyshev Type II order selection.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::real_vector_to_row_value;
use crate::builtins::math::signal::order_selection::{
    chebyshev_v_pass_stop, checked_order, classify_kind, db_excess_log_ratio,
    find_natural_frequency, option_text, parse_positive_db, real_edges, transform_edges,
    unwarp_frequency, validate_critical_frequencies, FilterKind, OrderFamily,
};
use crate::builtins::math::signal::type_resolvers::cheb2ord_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "cheb2ord";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::cheb2ord")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cheb2ord",
    op_kind: GpuOpKind::Custom("chebyshev2-order-selection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Chebyshev Type II order selection is scalar host-side analysis; GPU inputs are gathered automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::cheb2ord")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cheb2ord",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "cheb2ord materialises scalar/vector design parameters and is not fused.",
};

const CHEB2ORD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum Chebyshev Type II filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Natural cutoff frequency or frequency pair for Chebyshev Type II design.",
    },
];

const CHEB2ORD_INPUTS_DIGITAL: [BuiltinParamDescriptor; 4] = [
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

const CHEB2ORD_INPUTS_ANALOG: [BuiltinParamDescriptor; 5] = [
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

const CHEB2ORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs)",
        inputs: &CHEB2ORD_INPUTS_DIGITAL,
        outputs: &CHEB2ORD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs, 's')",
        inputs: &CHEB2ORD_INPUTS_ANALOG,
        outputs: &CHEB2ORD_OUTPUTS,
    },
];

const CHEB2ORD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.ARG_COUNT",
    identifier: Some("RunMat:cheb2ord:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "cheb2ord: expected cheb2ord(Wp, Ws, Rp, Rs [, 's'])",
};

const CHEB2ORD_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_FREQUENCY",
    identifier: Some("RunMat:cheb2ord:InvalidFrequency"),
    when: "Passband or stopband edges are invalid.",
    message: "cheb2ord: invalid passband or stopband frequency",
};

const CHEB2ORD_ERROR_INVALID_ATTENUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_ATTENUATION",
    identifier: Some("RunMat:cheb2ord:InvalidAttenuation"),
    when: "Ripple or attenuation values are invalid.",
    message: "cheb2ord: Rp and Rs must be positive finite scalars with Rs > Rp",
};

const CHEB2ORD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_OPTION",
    identifier: Some("RunMat:cheb2ord:InvalidOption"),
    when: "The optional selector is not supported.",
    message: "cheb2ord: optional selector must be 's'",
};

const CHEB2ORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:cheb2ord:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "cheb2ord: too many output arguments",
};

const CHEB2ORD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INTERNAL",
    identifier: Some("RunMat:cheb2ord:Internal"),
    when: "Output tensor construction fails internally.",
    message: "cheb2ord: internal error",
};

const CHEB2ORD_ERRORS: [BuiltinErrorDescriptor; 6] = [
    CHEB2ORD_ERROR_ARG_COUNT,
    CHEB2ORD_ERROR_INVALID_FREQUENCY,
    CHEB2ORD_ERROR_INVALID_ATTENUATION,
    CHEB2ORD_ERROR_INVALID_OPTION,
    CHEB2ORD_ERROR_TOO_MANY_OUTPUTS,
    CHEB2ORD_ERROR_INTERNAL,
];

pub const CHEB2ORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHEB2ORD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHEB2ORD_ERRORS,
};

#[derive(Clone, Debug)]
struct Cheb2ordResult {
    order: usize,
    wn: Vec<f64>,
}

fn cheb2ord_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cheb2ord_error_with_message(error.message, error)
}

fn cheb2ord_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cheb2ord_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cheb2ord_error_with_message(
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
    name = "cheb2ord",
    category = "math/signal",
    summary = "Select the minimum Chebyshev Type II filter order.",
    keywords = "cheb2ord,chebyshev,type II,filter order,signal processing",
    type_resolver(cheb2ord_type),
    descriptor(crate::builtins::math::signal::cheb2ord::CHEB2ORD_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::cheb2ord"
)]
async fn cheb2ord_builtin(
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
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_ARG_COUNT));
    }
    let analog = if let Some(option) = rest.first() {
        match option_text(option).as_deref() {
            Some("s") => true,
            Some(_) => return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_OPTION)),
            None => return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_OPTION)),
        }
    } else {
        false
    };
    let wp = real_edges(BUILTIN_NAME, "Wp", wp)
        .await
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    let ws = real_edges(BUILTIN_NAME, "Ws", ws)
        .await
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    let rp = parse_positive_db(BUILTIN_NAME, "Rp", &rp).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let rs = parse_positive_db(BUILTIN_NAME, "Rs", &rs).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    if rs <= rp {
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_INVALID_ATTENUATION));
    }

    let result = compute_cheb2ord(&wp, &ws, rp, rs, analog)?;
    output_result(result)
}

fn output_result(result: Cheb2ordResult) -> BuiltinResult<Value> {
    let order = Value::Num(result.order as f64);
    let wn = if result.wn.len() == 1 {
        Value::Num(result.wn[0])
    } else {
        real_vector_to_row_value(result.wn)
            .map_err(|err| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INTERNAL, err.message()))?
    };
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![order]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![order, wn]));
        }
        return Err(cheb2ord_error(&CHEB2ORD_ERROR_TOO_MANY_OUTPUTS));
    }
    Ok(order)
}

fn compute_cheb2ord(
    wp: &[f64],
    ws: &[f64],
    rp: f64,
    rs: f64,
    analog: bool,
) -> BuiltinResult<Cheb2ordResult> {
    if wp.len() != ws.len() {
        return Err(cheb2ord_error_with_detail(
            &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            "Wp and Ws must have the same length",
        ));
    }
    let kind = classify_kind(wp, ws)
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(wp, analog, "Wp")
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(ws, analog, "Ws")
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;

    let wp_work = transform_edges(wp, analog);
    let ws_work = transform_edges(ws, analog);
    let (nat, passband) =
        find_natural_frequency(kind, &wp_work, &ws_work, rp, rs, OrderFamily::Chebyshev).map_err(
            |detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail),
        )?;

    let log_ratio = db_excess_log_ratio(rp, rs).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let v_pass_stop = chebyshev_v_pass_stop(log_ratio).map_err(|detail| {
        cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_ATTENUATION, detail)
    })?;
    let order = checked_order(v_pass_stop / nat.acosh())
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;

    let prototype_edge = 1.0 / (v_pass_stop / order as f64).cosh();
    if !prototype_edge.is_finite() || prototype_edge <= 0.0 {
        return Err(cheb2ord_error_with_detail(
            &CHEB2ORD_ERROR_INVALID_FREQUENCY,
            "computed prototype edge is not finite and positive",
        ));
    }
    let natural = natural_frequency(kind, &passband, prototype_edge)?;
    let wn: Vec<f64> = if analog {
        natural
    } else {
        natural.into_iter().map(unwarp_frequency).collect()
    };
    validate_critical_frequencies(&wn, analog)
        .map_err(|detail| cheb2ord_error_with_detail(&CHEB2ORD_ERROR_INVALID_FREQUENCY, detail))?;
    Ok(Cheb2ordResult { order, wn })
}

fn natural_frequency(
    kind: FilterKind,
    passband: &[f64],
    prototype_edge: f64,
) -> BuiltinResult<Vec<f64>> {
    match kind {
        FilterKind::Lowpass => Ok(vec![passband[0] / prototype_edge]),
        FilterKind::Highpass => Ok(vec![passband[0] * prototype_edge]),
        FilterKind::Bandpass => {
            let bandwidth = passband[1] - passband[0];
            let center_sq = passband[0] * passband[1];
            let disc = bandwidth * bandwidth / (4.0 * prototype_edge * prototype_edge) + center_sq;
            let root = disc.sqrt();
            let first = (passband[0] - passband[1]) / (2.0 * prototype_edge) + root;
            let second = center_sq / first;
            let mut edges = vec![first, second];
            edges.sort_by(|a, b| a.total_cmp(b));
            Ok(edges)
        }
        FilterKind::Bandstop => {
            let center_sq = passband[0] * passband[1];
            let disc = prototype_edge * prototype_edge * (passband[1] - passband[0]).powi(2) / 4.0
                + center_sq;
            let root = disc.sqrt();
            let first = prototype_edge * (passband[0] - passband[1]) / 2.0 + root;
            let second = center_sq / first;
            let mut edges = vec![first, second];
            edges.sort_by(|a, b| a.total_cmp(b));
            Ok(edges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntegerStorage, Tensor};

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

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let mut tensor = Tensor::new_integer(storage, shape).expect("typed integer tensor");
        tensor.data.fill(f64::NAN);
        tensor
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("cheb2ord builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs)"));
    }

    #[test]
    fn production_highpass_repro_resolves() {
        let out = call(
            Value::Num(5000.0 / (20000.0 / 2.0)),
            Value::Num(4700.0 / (20000.0 / 2.0)),
            Value::Num(1.0),
            Value::Num(25.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 10.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.472_175_327_294_252_1).abs() < 1e-12);
    }

    #[test]
    fn digital_lowpass_and_highpass_return_stopband_side_edges() {
        let low = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(low);
        assert_eq!(n, 6.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 0.2);
        assert!(wn[0] < 0.3);

        let high = call(
            Value::Num(0.5),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(high);
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 0.3);
        assert!(wn[0] < 0.5);
    }

    #[test]
    fn bandpass_and_bandstop_return_two_edges() {
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

        let wp = Tensor::new(vec![0.1, 0.6], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.2, 0.5], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(3.0),
            Value::Num(60.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 7.0);
        assert_eq!(wn.len(), 2);
        assert!((wn[0] - 0.197_768_738_814_104_7).abs() < 1e-12);
        assert!((wn[1] - 0.503_814_419_296_416_3).abs() < 1e-12);
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
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 20.0);
    }

    #[test]
    fn analog_lowpass_reads_typed_integer_design_args_exactly() {
        let out = call(
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![10]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![20]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![40]), vec![1, 1])),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 5.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 20.0);
    }

    #[test]
    fn output_count_and_invalid_specs() {
        let one = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(1),
        )
        .unwrap();
        let Value::OutputList(values) = one else {
            panic!("expected output list");
        };
        assert_eq!(values.len(), 1);

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(3),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:TooManyOutputs"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(40.0),
            Value::Num(1.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidAttenuation"));

        let err = call(
            Value::Num(1.2),
            Value::Num(1.4),
            Value::Num(1.0),
            Value::Num(20.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidFrequency"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(20.0),
            &[Value::from("z")],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidOption"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.2000001),
            Value::Num(1.0),
            Value::Num(1.0e6),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:cheb2ord:InvalidFrequency"));
    }
}
