//! Focused MATLAB-compatible `freqz` digital filter response.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    parse_nonnegative_integer, parse_scalar_f64, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::freqz_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "freqz";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "freqz",
    op_kind: GpuOpKind::Custom("frequency-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Frequency-response analysis is evaluated on the host; GPU coefficient inputs are gathered automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "freqz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "freqz materialises response vectors and is not fused.",
};

const FREQZ_OUTPUT_H: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex frequency response.",
}];

const FREQZ_OUTPUT_H_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex frequency response.",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequencies in radians/sample or Hz when fs is supplied.",
    },
];

const FREQZ_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
];

const FREQZ_INPUTS_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
];

const FREQZ_INPUTS_N_FS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
    BuiltinParamDescriptor {
        name: "fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency for output frequencies in Hz.",
    },
];

const FREQZ_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H_W,
    },
];

const FREQZ_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.ARG_COUNT",
    identifier: Some("RunMat:freqz:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "freqz: expected freqz(b, a, [n, [fs]])",
};

const FREQZ_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:freqz:InvalidCoefficients"),
    when: "Coefficient inputs are empty or not numeric vectors.",
    message: "freqz: invalid coefficient input",
};

const FREQZ_ERROR_INVALID_N: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_N",
    identifier: Some("RunMat:freqz:InvalidN"),
    when: "The response length is not a positive integer scalar.",
    message: "freqz: n must be a positive integer",
};

const FREQZ_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_FS",
    identifier: Some("RunMat:freqz:InvalidFs"),
    when: "The sampling frequency is not a positive finite scalar.",
    message: "freqz: fs must be a positive finite scalar",
};

const FREQZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INTERNAL",
    identifier: Some("RunMat:freqz:Internal"),
    when: "Response tensor construction fails internally.",
    message: "freqz: internal error",
};

const FREQZ_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FREQZ_ERROR_ARG_COUNT,
    FREQZ_ERROR_INVALID_COEFFICIENTS,
    FREQZ_ERROR_INVALID_N,
    FREQZ_ERROR_INVALID_FS,
    FREQZ_ERROR_INTERNAL,
];

pub const FREQZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREQZ_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREQZ_ERRORS,
};

fn freqz_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    freqz_error_with_message(error.message, error)
}

fn freqz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    freqz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn freqz_error_with_message(
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
    name = "freqz",
    category = "math/signal",
    summary = "Evaluate digital filter frequency response.",
    keywords = "freqz,frequency response,filter,FIR,IIR,signal processing",
    type_resolver(freqz_type),
    descriptor(crate::builtins::math::signal::freqz::FREQZ_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::freqz"
)]
async fn freqz_builtin(b: Value, a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(b, a, &rest).await
}

pub async fn evaluate(b: Value, a: Value, rest: &[Value]) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(freqz_error(&FREQZ_ERROR_ARG_COUNT));
    }
    let b = value_to_complex_vector(BUILTIN_NAME, "numerator", b)
        .await
        .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_COEFFICIENTS, err.message()))?
        .data;
    let a = value_to_complex_vector(BUILTIN_NAME, "denominator", a)
        .await
        .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_COEFFICIENTS, err.message()))?
        .data;
    if b.is_empty() || a.is_empty() {
        return Err(freqz_error_with_detail(
            &FREQZ_ERROR_INVALID_COEFFICIENTS,
            "coefficient vectors cannot be empty",
        ));
    }
    let n = if let Some(value) = rest.first() {
        let parsed = parse_nonnegative_integer(BUILTIN_NAME, "n", value)
            .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_N, err.message()))?;
        if parsed == 0 {
            return Err(freqz_error(&FREQZ_ERROR_INVALID_N));
        }
        parsed
    } else {
        512
    };
    let fs = if let Some(value) = rest.get(1) {
        let fs = parse_scalar_f64(BUILTIN_NAME, "fs", value)
            .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_FS, err.message()))?;
        if fs <= 0.0 {
            return Err(freqz_error(&FREQZ_ERROR_INVALID_FS));
        }
        Some(fs)
    } else {
        None
    };

    let eval = evaluate_response(&b, &a, n, fs)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.h_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.h_value()?, eval.w_value()?],
        ));
    }
    eval.h_value()
}

struct FreqzEvaluation {
    h: Vec<Complex<f64>>,
    w: Vec<f64>,
}

impl FreqzEvaluation {
    fn h_value(&self) -> BuiltinResult<Value> {
        ComplexTensor::new(
            self.h.iter().map(|z| (z.re, z.im)).collect(),
            vec![self.h.len(), 1],
        )
        .map(Value::ComplexTensor)
        .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }

    fn w_value(&self) -> BuiltinResult<Value> {
        Tensor::new(self.w.clone(), vec![self.w.len(), 1])
            .map(Value::Tensor)
            .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }
}

fn evaluate_response(
    b: &[Complex<f64>],
    a: &[Complex<f64>],
    n: usize,
    fs: Option<f64>,
) -> BuiltinResult<FreqzEvaluation> {
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for idx in 0..n {
        let omega = std::f64::consts::PI * idx as f64 / n as f64;
        let point = Complex::new(omega.cos(), -omega.sin());
        let numerator = polynomial_in_z_inverse(b, point);
        let denominator = polynomial_in_z_inverse(a, point);
        h.push(numerator / denominator);
        w.push(match fs {
            Some(fs) => fs * idx as f64 / (2.0 * n as f64),
            None => omega,
        });
    }
    Ok(FreqzEvaluation { h, w })
}

fn polynomial_in_z_inverse(coeffs: &[Complex<f64>], point: Complex<f64>) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    let mut power = Complex::new(1.0, 0.0);
    for coeff in coeffs {
        acc += *coeff * power;
        power *= point;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;

    fn call(b: Value, a: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(b, a, rest))
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("freqz builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[H, w] = freqz(b, a, n, fs)"));
    }

    #[test]
    fn simple_fir_response_matches_closed_form() {
        let h = call(
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .unwrap();
        let Value::ComplexTensor(h) = h else {
            panic!("expected complex response");
        };
        assert_eq!(h.shape, vec![4, 1]);
        assert!((h.data[0].0 - 2.0).abs() < 1e-12);
        assert!((h.data[2].0 - 1.0).abs() < 1e-12);
        assert!((h.data[2].1 + 1.0).abs() < 1e-12);
    }

    #[test]
    fn iir_response_and_frequency_outputs() {
        let out = call(
            Value::Num(0.2),
            Value::Tensor(Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap()),
            &[Value::Num(8.0), Value::Num(1000.0)],
            Some(2),
        )
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        let Value::ComplexTensor(h) = &values[0] else {
            panic!("expected H");
        };
        let Value::Tensor(w) = &values[1] else {
            panic!("expected w");
        };
        assert!((h.data[0].0 - 1.0).abs() < 1e-12);
        assert_eq!(w.shape, vec![8, 1]);
        assert!((w.data[1] - 62.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_n_and_empty_coefficients() {
        assert!(call(Value::Num(1.0), Value::Num(1.0), &[Value::Num(0.0)], None).is_err());
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        assert!(call(Value::Tensor(empty), Value::Num(1.0), &[], None).is_err());
    }
}
