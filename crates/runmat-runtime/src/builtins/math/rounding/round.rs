//! MATLAB-compatible `round` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::round")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "round",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_round" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute round directly on the device; digit-aware rounding currently gathers to the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::round")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "round",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("round({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `round` calls; providers can substitute custom kernels.",
};

const BUILTIN_NAME: &str = "round";

const ROUND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rounded output values.",
}];
const ROUND_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, or complex input values.",
}];
const ROUND_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, or complex input values.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Digits for decimal-place rounding.",
    },
];
const ROUND_INPUTS_X_N_MODE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, or complex input values.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Digits argument.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"decimals\""),
        description: "Rounding mode ('decimals' or 'significant').",
    },
];
const ROUND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = round(X)",
        inputs: &ROUND_INPUTS_X,
        outputs: &ROUND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = round(X, N)",
        inputs: &ROUND_INPUTS_X_N,
        outputs: &ROUND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = round(X, N, mode)",
        inputs: &ROUND_INPUTS_X_N_MODE,
        outputs: &ROUND_OUTPUT,
    },
];
const ROUND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_INPUT",
    identifier: Some("RunMat:round:InvalidInput"),
    when: "Input X cannot be interpreted as numeric/logical/complex data.",
    message: "round: invalid input",
};
const ROUND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_ARGUMENT",
    identifier: Some("RunMat:round:InvalidArgument"),
    when: "Argument count does not match supported call forms.",
    message: "round: invalid argument",
};
const ROUND_ERROR_INVALID_DIGITS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_DIGITS",
    identifier: Some("RunMat:round:InvalidDigits"),
    when: "N is not an integer scalar or violates mode constraints.",
    message: "round: invalid digits argument",
};
const ROUND_ERROR_INVALID_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_MODE",
    identifier: Some("RunMat:round:InvalidMode"),
    when: "mode is not a supported text token.",
    message: "round: invalid mode",
};
const ROUND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INTERNAL",
    identifier: Some("RunMat:round:Internal"),
    when: "Internal tensor conversion/allocation failed.",
    message: "round: internal error",
};
const ROUND_ERRORS: [BuiltinErrorDescriptor; 5] = [
    ROUND_ERROR_INVALID_INPUT,
    ROUND_ERROR_INVALID_ARGUMENT,
    ROUND_ERROR_INVALID_DIGITS,
    ROUND_ERROR_INVALID_MODE,
    ROUND_ERROR_INTERNAL,
];
pub const ROUND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ROUND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ROUND_ERRORS,
};

fn builtin_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundStrategy {
    Integer,
    Decimals(i32),
    Significant(i32),
}

impl RoundStrategy {
    fn requires_host(&self) -> bool {
        !matches!(self, RoundStrategy::Integer)
    }
}

#[runtime_builtin(
    name = "round",
    category = "math/rounding",
    summary = "Round values to the nearest integers, decimal places, or significant digits.",
    keywords = "round,rounding,significant,decimals,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    descriptor(crate::builtins::math::rounding::round::ROUND_DESCRIPTOR),
    builtin_path = "crate::builtins::math::rounding::round"
)]
async fn round_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let strategy = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => round_gpu(handle, strategy).await,
        Value::Complex(re, im) => Ok(Value::Complex(
            round_scalar(re, strategy),
            round_scalar(im, strategy),
        )),
        Value::ComplexTensor(ct) => round_complex_tensor(ct, strategy),
        Value::CharArray(ca) => round_char_array(ca, strategy),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| builtin_error_with_detail(&ROUND_ERROR_INVALID_INPUT, err))?;
            Ok(round_tensor(tensor, strategy).map(tensor::tensor_into_value)?)
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error_with_detail(
            &ROUND_ERROR_INVALID_INPUT,
            "expected numeric or logical input",
        )),
        other => round_numeric(other, strategy),
    }
}

async fn round_gpu(handle: GpuTensorHandle, strategy: RoundStrategy) -> BuiltinResult<Value> {
    if !strategy.requires_host() {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_round(&handle).await {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    round_tensor(tensor, strategy).map(tensor::tensor_into_value)
}

fn round_numeric(value: Value, strategy: RoundStrategy) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Num(round_scalar(n, strategy))),
        Value::Int(i) => Ok(Value::Num(round_scalar(i.to_f64(), strategy))),
        Value::Bool(b) => Ok(Value::Num(round_scalar(
            if b { 1.0 } else { 0.0 },
            strategy,
        ))),
        Value::Tensor(t) => round_tensor(t, strategy).map(tensor::tensor_into_value),
        other => {
            let tensor = tensor::value_into_tensor_for("round", other)
                .map_err(|err| builtin_error_with_detail(&ROUND_ERROR_INVALID_INPUT, err))?;
            Ok(round_tensor(tensor, strategy).map(tensor::tensor_into_value)?)
        }
    }
}

fn round_tensor(mut tensor: Tensor, strategy: RoundStrategy) -> BuiltinResult<Tensor> {
    for value in &mut tensor.data {
        *value = round_scalar(*value, strategy);
    }
    Ok(tensor)
}

fn round_complex_tensor(ct: ComplexTensor, strategy: RoundStrategy) -> BuiltinResult<Value> {
    let data = ct
        .data
        .iter()
        .map(|&(re, im)| (round_scalar(re, strategy), round_scalar(im, strategy)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error_with_detail(&ROUND_ERROR_INTERNAL, e))?;
    Ok(Value::ComplexTensor(tensor))
}

fn round_char_array(ca: CharArray, strategy: RoundStrategy) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(round_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error_with_detail(&ROUND_ERROR_INTERNAL, e))?;
    Ok(Value::Tensor(tensor))
}

fn round_scalar(value: f64, strategy: RoundStrategy) -> f64 {
    if !value.is_finite() {
        return value;
    }
    match strategy {
        RoundStrategy::Integer => value.round(),
        RoundStrategy::Decimals(n) => round_with_decimals(value, n),
        RoundStrategy::Significant(n) => round_with_significant(value, n),
    }
}

fn round_with_decimals(value: f64, digits: i32) -> f64 {
    if digits == 0 {
        return value.round();
    }
    let factor = 10f64.powi(digits);
    if !factor.is_finite() || factor == 0.0 {
        // Large magnitude digits saturate: rounding has no effect.
        return value;
    }
    (value * factor).round() / factor
}

fn round_with_significant(value: f64, digits: i32) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    let abs_val = value.abs();
    let order = abs_val.log10().floor();
    let scale_power = digits - 1 - order as i32;
    let scale = 10f64.powi(scale_power);
    if !scale.is_finite() || scale == 0.0 {
        return value;
    }
    (value * scale).round() / scale
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<RoundStrategy> {
    match args.len() {
        0 => Ok(RoundStrategy::Integer),
        1 => {
            let digits = parse_digits(&args[0])?;
            Ok(RoundStrategy::Decimals(digits))
        }
        2 => {
            let digits = parse_digits(&args[0])?;
            let mode = parse_mode(&args[1])?;
            match mode {
                RoundMode::Decimals => Ok(RoundStrategy::Decimals(digits)),
                RoundMode::Significant => {
                    if digits <= 0 {
                        return Err(builtin_error_with_detail(
                            &ROUND_ERROR_INVALID_DIGITS,
                            "N must be a positive integer for 'significant' rounding",
                        ));
                    }
                    Ok(RoundStrategy::Significant(digits))
                }
            }
        }
        _ => Err(builtin_error_with_detail(
            &ROUND_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
    }
}

fn parse_digits(value: &Value) -> BuiltinResult<i32> {
    let err =
        || builtin_error_with_detail(&ROUND_ERROR_INVALID_DIGITS, "N must be an integer scalar");
    let raw = match value {
        Value::Int(i) => i.to_i64(),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(err());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(err());
            }
            rounded as i64
        }
        Value::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        other => {
            return Err(builtin_error_with_detail(
                &ROUND_ERROR_INVALID_DIGITS,
                format!("N must be numeric, got {:?}", other),
            ))
        }
    };
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err(builtin_error_with_detail(
            &ROUND_ERROR_INVALID_DIGITS,
            "integer overflow in N",
        ));
    }
    Ok(raw as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundMode {
    Decimals,
    Significant,
}

fn parse_mode(value: &Value) -> BuiltinResult<RoundMode> {
    let Some(text) = tensor::value_to_string(value) else {
        return Err(builtin_error_with_detail(
            &ROUND_ERROR_INVALID_MODE,
            "mode must be a character vector or string scalar",
        ));
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(RoundMode::Significant),
        "decimal" | "decimals" => Ok(RoundMode::Decimals),
        other => Err(builtin_error_with_detail(
            &ROUND_ERROR_INVALID_MODE,
            format!("unknown rounding mode '{other}'"),
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, ResolveContext, Tensor, Type};

    fn round_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::round_builtin(value, rest))
    }

    fn assert_error_contains(err: &crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn round_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ROUND_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = round(X)"));
        assert!(labels.contains(&"Y = round(X, N)"));
        assert!(labels.contains(&"Y = round(X, N, mode)"));
    }

    #[test]
    fn round_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn round_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_scalar_defaults() {
        let result = round_builtin(Value::Num(1.7), Vec::new()).expect("round");
        match result {
            Value::Num(v) => assert_eq!(v, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_scalar_negative_half() {
        let result = round_builtin(Value::Num(-2.5), Vec::new()).expect("round");
        match result {
            Value::Num(v) => assert_eq!(v, -3.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_tensor_decimals() {
        let tensor = Tensor::new(vec![1.2345, 2.499, 3.5001], vec![3, 1]).unwrap();
        let result = round_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))])
            .expect("round");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [1.23, 2.5, 3.5];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_tensor_negative_decimals() {
        let tensor = Tensor::new(vec![123.0, 149.9, 150.0], vec![3, 1]).unwrap();
        let result = round_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(-2))])
            .expect("round");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![100.0, 100.0, 200.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_scalar_significant() {
        let result = round_builtin(
            Value::Num(0.0012345),
            vec![Value::Int(IntValue::I32(3)), Value::from("significant")],
        )
        .expect("round");
        match result {
            Value::Num(v) => assert!((v - 0.00123).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_complex_value() {
        let result = round_builtin(Value::Complex(1.2, -3.6), Vec::new()).expect("round");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -4.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_invalid_mode_errors() {
        let err = round_builtin(
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(2)), Value::from("approx")],
        )
        .unwrap_err();
        assert_error_contains(&err, "unknown rounding mode");
        assert_eq!(err.identifier(), ROUND_ERROR_INVALID_MODE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn round_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.5, -0.2, 0.5, 1.8], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = round_builtin(Value::GpuTensor(handle), Vec::new()).expect("round");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![-3.0, 0.0, 1.0, 2.0]);
        });
    }
}
