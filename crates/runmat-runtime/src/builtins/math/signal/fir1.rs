//! Focused MATLAB-compatible `fir1` FIR-window design.

use num_complex::Complex;
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
    keyword, parse_nonnegative_integer, real_vector_to_row_value, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::fir1_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fir1";
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::fir1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fir1",
    op_kind: GpuOpKind::Custom("fir-design"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Coefficient design is host-side; generated coefficients can be used with GPU-aware filtering builtins.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::fir1")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fir1",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fir1 materialises coefficient vectors and is not fused.",
};

const FIR1_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FIR numerator coefficient row vector.",
}];

const FIR1_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
];

const FIR1_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Filter type, window vector, or scale/noscale option.",
    },
];

const FIR1_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn)",
        inputs: &FIR1_INPUTS_CORE,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype, window, scaleopt)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
];

const FIR1_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.ARG_COUNT",
    identifier: Some("RunMat:fir1:ArgCount"),
    when: "Fewer than two arguments are supplied.",
    message: "fir1: expected fir1(n, Wn, ...)",
};

const FIR1_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_ORDER",
    identifier: Some("RunMat:fir1:InvalidOrder"),
    when: "The order is not a finite nonnegative integer scalar.",
    message: "fir1: filter order must be a nonnegative integer",
};

const FIR1_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_FREQUENCY",
    identifier: Some("RunMat:fir1:InvalidFrequency"),
    when: "Wn is not a valid normalized scalar or increasing two-element band.",
    message: "fir1: invalid cutoff frequency",
};

const FIR1_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_OPTION",
    identifier: Some("RunMat:fir1:InvalidOption"),
    when: "A filter type, window vector, or scaling option is unsupported.",
    message: "fir1: invalid option",
};

const FIR1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INTERNAL",
    identifier: Some("RunMat:fir1:Internal"),
    when: "Coefficient materialization fails internally.",
    message: "fir1: internal error",
};

const FIR1_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FIR1_ERROR_ARG_COUNT,
    FIR1_ERROR_INVALID_ORDER,
    FIR1_ERROR_INVALID_FREQUENCY,
    FIR1_ERROR_INVALID_OPTION,
    FIR1_ERROR_INTERNAL,
];

pub const FIR1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIR1_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIR1_ERRORS,
};

fn fir1_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fir1_error_with_message(error.message, error)
}

fn fir1_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fir1_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fir1_error_with_message(
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
    name = "fir1",
    category = "math/signal",
    summary = "Design windowed-sinc FIR filters.",
    keywords = "fir1,FIR,windowed sinc,lowpass,highpass,bandpass,bandstop,signal processing",
    type_resolver(fir1_type),
    descriptor(crate::builtins::math::signal::fir1::FIR1_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::fir1"
)]
async fn fir1_builtin(n: Value, wn: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(n, wn, &rest).await
}

pub async fn evaluate(n: Value, wn: Value, rest: &[Value]) -> BuiltinResult<Value> {
    let requested_order = parse_nonnegative_integer(BUILTIN_NAME, "order", &n)
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_ORDER, err.message()))?;
    let cutoff = parse_cutoff(wn).await?;
    let mut options = Fir1Options::default_for_cutoff(&cutoff);
    let mut window: Option<Value> = None;

    for arg in rest {
        if let Some(word) = keyword(arg) {
            match word.as_str() {
                "low" | "lowpass" => options.kind = FilterKind::Lowpass,
                "high" | "highpass" => options.kind = FilterKind::Highpass,
                "bandpass" | "pass" => options.kind = FilterKind::Bandpass,
                "stop" | "bandstop" | "bandreject" => options.kind = FilterKind::Bandstop,
                "scale" => options.scale = true,
                "noscale" => options.scale = false,
                _ => {
                    return Err(fir1_error_with_detail(
                        &FIR1_ERROR_INVALID_OPTION,
                        format!("unknown option '{word}'"),
                    ))
                }
            }
        } else {
            if window.is_some() {
                return Err(fir1_error_with_detail(
                    &FIR1_ERROR_INVALID_OPTION,
                    "multiple window vectors supplied",
                ));
            }
            window = Some(arg.clone());
        }
    }

    validate_kind_cutoff(options.kind, &cutoff)?;
    let order = adjusted_order(requested_order, options.kind);
    let window = if let Some(window) = window {
        parse_window(window, order + 1).await?
    } else {
        hamming_window(order + 1)
    };
    let mut coeffs = ideal_impulse_response(order, &cutoff, options.kind);
    for (coeff, win) in coeffs.iter_mut().zip(window.iter()) {
        *coeff *= *win;
    }
    if options.scale {
        scale_coefficients(&mut coeffs, &cutoff, options.kind)?;
    }
    real_vector_to_row_value(coeffs)
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INTERNAL, err.message()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Clone, Copy)]
struct Fir1Options {
    kind: FilterKind,
    scale: bool,
}

impl Fir1Options {
    fn default_for_cutoff(cutoff: &[f64]) -> Self {
        Self {
            kind: if cutoff.len() == 1 {
                FilterKind::Lowpass
            } else {
                FilterKind::Bandpass
            },
            scale: true,
        }
    }
}

async fn parse_cutoff(value: Value) -> BuiltinResult<Vec<f64>> {
    let input = value_to_complex_vector(BUILTIN_NAME, "Wn", value)
        .await
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_FREQUENCY, err.message()))?;
    if input.data.is_empty() || input.data.len() > 2 || input.data.iter().any(|z| z.im.abs() > EPS)
    {
        return Err(fir1_error(&FIR1_ERROR_INVALID_FREQUENCY));
    }
    let cutoff = input.data.iter().map(|z| z.re).collect::<Vec<_>>();
    if cutoff
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
        || (cutoff.len() == 2 && cutoff[0] >= cutoff[1])
    {
        return Err(fir1_error(&FIR1_ERROR_INVALID_FREQUENCY));
    }
    Ok(cutoff)
}

async fn parse_window(value: Value, expected_len: usize) -> BuiltinResult<Vec<f64>> {
    let input = value_to_complex_vector(BUILTIN_NAME, "window", value)
        .await
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_OPTION, err.message()))?;
    if input.data.len() != expected_len || input.data.iter().any(|z| z.im.abs() > EPS) {
        return Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            format!("window must be a real vector of length {expected_len}"),
        ));
    }
    Ok(input.data.iter().map(|z| z.re).collect())
}

fn validate_kind_cutoff(kind: FilterKind, cutoff: &[f64]) -> BuiltinResult<()> {
    match (kind, cutoff.len()) {
        (FilterKind::Lowpass | FilterKind::Highpass, 1)
        | (FilterKind::Bandpass | FilterKind::Bandstop, 2) => Ok(()),
        _ => Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            "filter type is incompatible with Wn shape",
        )),
    }
}

fn adjusted_order(order: usize, kind: FilterKind) -> usize {
    match kind {
        FilterKind::Highpass | FilterKind::Bandstop if order % 2 == 1 => order + 1,
        _ => order,
    }
}

fn hamming_window(len: usize) -> Vec<f64> {
    match len {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..len)
            .map(|idx| {
                let phase = 2.0 * std::f64::consts::PI * idx as f64 / (len - 1) as f64;
                0.54 - 0.46 * phase.cos()
            })
            .collect(),
    }
}

fn ideal_impulse_response(order: usize, cutoff: &[f64], kind: FilterKind) -> Vec<f64> {
    let center = order as f64 / 2.0;
    (0..=order)
        .map(|idx| {
            let m = idx as f64 - center;
            match kind {
                FilterKind::Lowpass => lowpass_sample(cutoff[0], m),
                FilterKind::Highpass => delta(m) - lowpass_sample(cutoff[0], m),
                FilterKind::Bandpass => lowpass_sample(cutoff[1], m) - lowpass_sample(cutoff[0], m),
                FilterKind::Bandstop => {
                    delta(m) - (lowpass_sample(cutoff[1], m) - lowpass_sample(cutoff[0], m))
                }
            }
        })
        .collect()
}

fn lowpass_sample(cutoff: f64, m: f64) -> f64 {
    cutoff * sinc(cutoff * m)
}

fn sinc(x: f64) -> f64 {
    if x.abs() <= EPS {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn delta(m: f64) -> f64 {
    if m.abs() <= EPS {
        1.0
    } else {
        0.0
    }
}

fn scale_coefficients(coeffs: &mut [f64], cutoff: &[f64], kind: FilterKind) -> BuiltinResult<()> {
    let omega = match kind {
        FilterKind::Lowpass | FilterKind::Bandstop => 0.0,
        FilterKind::Highpass => std::f64::consts::PI,
        FilterKind::Bandpass => std::f64::consts::PI * (cutoff[0] + cutoff[1]) / 2.0,
    };
    let response = frequency_response(coeffs, omega).norm();
    if response <= EPS || !response.is_finite() {
        return Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            "cannot scale filter with near-zero passband response",
        ));
    }
    for coeff in coeffs {
        *coeff /= response;
    }
    Ok(())
}

fn frequency_response(coeffs: &[f64], omega: f64) -> Complex<f64> {
    coeffs
        .iter()
        .enumerate()
        .fold(Complex::new(0.0, 0.0), |acc, (idx, coeff)| {
            let phase = -omega * idx as f64;
            acc + Complex::new(phase.cos(), phase.sin()) * *coeff
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, Tensor};

    fn call(n: Value, wn: Value, rest: &[Value]) -> BuiltinResult<Value> {
        block_on(evaluate(n, wn, rest))
    }

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fir1 builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "b = fir1(n, Wn)"));
    }

    #[test]
    fn lowpass_has_expected_shape_symmetry_and_dc_gain() {
        let out = tensor(call(Value::Num(10.0), Value::Num(0.25), &[]).unwrap());
        assert_eq!(out.shape, vec![1, 11]);
        for idx in 0..out.data.len() {
            let mirror = out.data.len() - 1 - idx;
            assert!((out.data[idx] - out.data[mirror]).abs() < 1e-12);
        }
        let sum: f64 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn highpass_scales_at_nyquist() {
        let out = tensor(call(Value::Num(10.0), Value::Num(0.35), &[Value::from("high")]).unwrap());
        let response = frequency_response(&out.data, std::f64::consts::PI).norm();
        assert!((response - 1.0).abs() < 1e-10);
    }

    #[test]
    fn odd_order_highpass_is_adjusted_to_even_order() {
        let out = tensor(call(Value::Num(5.0), Value::Num(0.35), &[Value::from("high")]).unwrap());
        assert_eq!(out.shape, vec![1, 7]);
        let response = frequency_response(&out.data, std::f64::consts::PI).norm();
        assert!((response - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bandpass_and_stop_accept_two_element_cutoff() {
        let wn = Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap();
        let bandpass = tensor(call(Value::Num(20.0), Value::Tensor(wn.clone()), &[]).unwrap());
        assert_eq!(bandpass.shape, vec![1, 21]);

        let stop =
            tensor(call(Value::Num(20.0), Value::Tensor(wn), &[Value::from("stop")]).unwrap());
        let dc: f64 = stop.data.iter().sum();
        assert!((dc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn custom_window_and_noscale_are_supported() {
        let window = Tensor::new(vec![1.0; 5], vec![1, 5]).unwrap();
        let out = tensor(
            call(
                Value::Num(4.0),
                Value::Num(0.4),
                &[Value::Tensor(window), Value::from("noscale")],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![1, 5]);
        assert!((out.data.iter().sum::<f64>() - 1.0).abs() > 1e-3);
    }

    #[test]
    fn rejects_invalid_cutoff_and_window() {
        assert!(call(Value::Num(4.0), Value::Num(1.2), &[]).is_err());
        let window = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        assert!(call(Value::Num(4.0), Value::Num(0.4), &[Value::Tensor(window)]).is_err());
    }
}
