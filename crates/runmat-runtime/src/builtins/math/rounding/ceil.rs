//! MATLAB-compatible `ceil` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::ceil")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ceil",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_ceil" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute ceil directly on the device; the runtime gathers to the host when unary_ceil is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::ceil")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ceil",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("ceil({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `ceil` calls; providers can substitute custom kernels when available.",
};

const BUILTIN_NAME: &str = "ceil";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ceil",
    category = "math/rounding",
    summary = "Round values toward positive infinity.",
    keywords = "ceil,rounding,integers,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::rounding::ceil"
)]
async fn ceil_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_arguments(&rest).await?;
    let base = match value {
        Value::GpuTensor(handle) => ceil_gpu(handle, &args).await?,
        Value::Complex(re, im) => Value::Complex(
            apply_ceil_scalar(re, args.strategy),
            apply_ceil_scalar(im, args.strategy),
        ),
        Value::ComplexTensor(ct) => ceil_complex_tensor(ct, args.strategy)?,
        Value::CharArray(ca) => ceil_char_array(ca, args.strategy)?,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            let ceiled = ceil_tensor(tensor, args.strategy)?;
            tensor::tensor_into_value(ceiled)
        }
        Value::String(_) | Value::StringArray(_) => {
            return Err(builtin_error("ceil: expected numeric or logical input"));
        }
        other => ceil_numeric(other, args.strategy)?,
    };
    apply_output_template(base, &args.output).await
}

fn ceil_numeric(value: Value, strategy: CeilStrategy) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("ceil", value).map_err(|err| builtin_error(err))?;
    let ceiled = ceil_tensor(tensor, strategy)?;
    Ok(tensor::tensor_into_value(ceiled))
}

fn ceil_tensor(mut tensor: Tensor, strategy: CeilStrategy) -> BuiltinResult<Tensor> {
    for value in &mut tensor.data {
        *value = apply_ceil_scalar(*value, strategy);
    }
    Ok(tensor)
}

fn ceil_complex_tensor(ct: ComplexTensor, strategy: CeilStrategy) -> BuiltinResult<Value> {
    let data: Vec<(f64, f64)> = ct
        .data
        .iter()
        .map(|&(re, im)| {
            (
                apply_ceil_scalar(re, strategy),
                apply_ceil_scalar(im, strategy),
            )
        })
        .collect();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error(format!("ceil: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn ceil_char_array(ca: CharArray, strategy: CeilStrategy) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(apply_ceil_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("ceil: {e}")))?;
    Ok(Value::Tensor(tensor))
}

async fn ceil_gpu(handle: GpuTensorHandle, args: &CeilArgs) -> BuiltinResult<Value> {
    if matches!(args.strategy, CeilStrategy::Integer) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_ceil(&handle).await {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let ceiled = ceil_tensor(tensor, args.strategy)?;
    Ok(tensor::tensor_into_value(ceiled))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CeilStrategy {
    Integer,
    Decimals(i32),
    Significant(i32),
}

#[derive(Clone, Debug)]
struct CeilArgs {
    strategy: CeilStrategy,
    output: OutputTemplate,
}

#[derive(Clone, Debug)]
enum OutputTemplate {
    Default,
    Like(Value),
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<CeilArgs> {
    let (strategy_len, output) = parse_output_template(args)?;
    let strategy = match strategy_len {
        0 => CeilStrategy::Integer,
        1 => CeilStrategy::Decimals(parse_digits(&args[0]).await?),
        2 => {
            let digits = parse_digits(&args[0]).await?;
            let mode = parse_mode(&args[1])?;
            match mode {
                CeilMode::Decimals => CeilStrategy::Decimals(digits),
                CeilMode::Significant => {
                    if digits <= 0 {
                        return Err(builtin_error(
                            "ceil: N must be a positive integer for 'significant' rounding",
                        ));
                    }
                    CeilStrategy::Significant(digits)
                }
            }
        }
        _ => return Err(builtin_error("ceil: too many input arguments")),
    };
    Ok(CeilArgs { strategy, output })
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<(usize, OutputTemplate)> {
    if !args.is_empty() && is_keyword(&args[args.len() - 1], "like") {
        return Err(builtin_error("ceil: expected prototype after 'like'"));
    }
    if args.len() >= 2 && is_keyword(&args[args.len() - 2], "like") {
        let proto = &args[args.len() - 1];
        if matches!(
            proto,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err(builtin_error("ceil: unsupported prototype for 'like'"));
        }
        return Ok((args.len() - 2, OutputTemplate::Like(proto.clone())));
    }
    Ok((args.len(), OutputTemplate::Default))
}

async fn parse_digits(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle.clone());
            let gathered = gpu_helpers::gather_value_async(&proxy).await?;
            parse_digits_inner(&gathered)
        }
        other => parse_digits_inner(other),
    }
}

fn parse_digits_inner(value: &Value) -> BuiltinResult<i32> {
    const ERR: &str = "ceil: N must be an integer scalar";
    let raw = match value {
        Value::Int(i) => i.to_i64(),
        Value::Num(n) => return digits_from_f64(*n),
        Value::Bool(b) => {
            if *b {
                1
            } else {
                0
            }
        }
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(builtin_error(ERR));
            }
            return digits_from_f64(tensor.data[0]);
        }
        Value::LogicalArray(logical) => {
            if logical.len() != 1 {
                return Err(builtin_error(ERR));
            }
            if logical.data[0] != 0 {
                1
            } else {
                0
            }
        }
        other => {
            return Err(builtin_error(format!(
                "ceil: N must be numeric, got {:?}",
                other
            )))
        }
    };
    digits_from_i64(raw)
}

fn digits_from_f64(value: f64) -> BuiltinResult<i32> {
    if !value.is_finite() {
        return Err(builtin_error("ceil: N must be an integer scalar"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(builtin_error("ceil: N must be an integer scalar"));
    }
    if rounded > i64::MAX as f64 || rounded < i64::MIN as f64 {
        return Err(builtin_error("ceil: integer overflow in N"));
    }
    digits_from_i64(rounded as i64)
}

fn digits_from_i64(raw: i64) -> BuiltinResult<i32> {
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err(builtin_error("ceil: integer overflow in N"));
    }
    Ok(raw as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CeilMode {
    Decimals,
    Significant,
}

fn parse_mode(value: &Value) -> BuiltinResult<CeilMode> {
    let Some(text) = tensor::value_to_string(value) else {
        return Err(builtin_error(
            "ceil: mode must be a character vector or string scalar",
        ));
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(CeilMode::Significant),
        "decimal" | "decimals" | "digits" | "places" | "place" => Ok(CeilMode::Decimals),
        other => Err(builtin_error(format!(
            "ceil: unknown rounding mode '{other}'"
        ))),
    }
}

fn is_keyword(value: &Value, target: &str) -> bool {
    tensor::value_to_string(value)
        .map(|s| s.trim().eq_ignore_ascii_case(target))
        .unwrap_or(false)
}

fn apply_ceil_scalar(value: f64, strategy: CeilStrategy) -> f64 {
    if !value.is_finite() {
        return value;
    }
    match strategy {
        CeilStrategy::Integer => value.ceil(),
        CeilStrategy::Decimals(digits) => ceil_with_decimals(value, digits),
        CeilStrategy::Significant(digits) => ceil_with_significant(value, digits),
    }
}

fn ceil_with_decimals(value: f64, digits: i32) -> f64 {
    if digits == 0 {
        return value.ceil();
    }
    let factor = 10f64.powi(digits);
    if !factor.is_finite() || factor == 0.0 {
        return value;
    }
    (value * factor).ceil() / factor
}

fn ceil_with_significant(value: f64, digits: i32) -> f64 {
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
    (value * scale).ceil() / scale
}

async fn apply_output_template(value: Value, output: &OutputTemplate) -> BuiltinResult<Value> {
    match output {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(_) => convert_to_gpu(value),
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_) => convert_to_host_like(value).await,
            _ => Err(builtin_error(
                "ceil: unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        builtin_error(
            "ceil: GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| builtin_error(format!("ceil: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor =
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("ceil: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        other => Err(builtin_error(format!(
            "ceil: 'like' GPU prototypes are only supported for real numeric outputs (got {other:?})"
        ))),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy).await
        }
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Tensor, Type, Value};

    fn ceil_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ceil_builtin(value, rest))
    }

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[test]
    fn ceil_type_preserves_tensor_shape() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn ceil_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_scalar_positive_and_negative() {
        let value = Value::Num(-2.7);
        let result = ceil_builtin(value, Vec::new()).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, -2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_integer_tensor() {
        let tensor = Tensor::new(vec![1.2, 4.7, -3.4, 5.0], vec![2, 2]).unwrap();
        let result = ceil_builtin(Value::Tensor(tensor), Vec::new()).expect("ceil");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![2.0, 5.0, -3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_complex_value() {
        let result = ceil_builtin(Value::Complex(1.7, -2.3), Vec::new()).expect("ceil");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 2.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_char_array_to_tensor() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = ceil_builtin(Value::CharArray(chars), Vec::new()).expect("ceil");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 66.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_logical_array_remains_same() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = ceil_builtin(Value::LogicalArray(logical), Vec::new()).expect("ceil");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 0.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_int_value_passthrough() {
        let result = ceil_builtin(Value::Int(IntValue::I32(-4)), Vec::new()).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, -4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.2, 1.9, -0.1, -3.8], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = ceil_builtin(Value::GpuTensor(handle), Vec::new()).expect("ceil");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 0.0, -3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_decimal_digits() {
        let value = Value::Num(21.456);
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = ceil_builtin(value, args).expect("ceil");
        match result {
            Value::Num(v) => assert!((v - 21.46).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_negative_digits() {
        let tensor = Tensor::new(vec![123.4, -987.6], vec![2, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-2))];
        let result = ceil_builtin(Value::Tensor(tensor), args).expect("ceil");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![200.0, -900.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_digits_accepts_tensor_scalar() {
        let value = Value::Tensor(Tensor::new(vec![1.234], vec![1, 1]).unwrap());
        let digits = Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap());
        let result = ceil_builtin(value, vec![digits]).expect("ceil");
        match result {
            Value::Num(v) => assert!((v - 1.24).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_digits_accepts_gpu_scalar() {
        test_support::with_test_provider(|provider| {
            let digits_tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &digits_tensor.data,
                shape: &digits_tensor.shape,
            };
            let digits_handle = provider.upload(&view).expect("upload digits");
            let args = vec![Value::GpuTensor(digits_handle)];
            let result = ceil_builtin(Value::Num(1.234), args).expect("ceil");
            match result {
                Value::Num(v) => assert!((v - 1.24).abs() < 1e-12),
                other => panic!("expected scalar result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_significant_digits() {
        let value = Value::Num(98765.4321);
        let args = vec![Value::Int(IntValue::I32(3)), Value::from("significant")];
        let result = ceil_builtin(value, args).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, 98800.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_significant_negative_numbers() {
        let value = Value::Num(-0.01234);
        let args = vec![Value::Int(IntValue::I32(2)), Value::from("significant")];
        let result = ceil_builtin(value, args).expect("ceil");
        match result {
            Value::Num(v) => assert!((v - -0.012).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_significant_requires_positive_digits() {
        let args = vec![Value::Int(IntValue::I32(0)), Value::from("significant")];
        let err = ceil_builtin(Value::Num(1.23), args).unwrap_err();
        assert_error_contains(err, "positive integer");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_decimal_mode_alias_digits_keyword() {
        let args = vec![Value::Int(IntValue::I32(1)), Value::from("digits")];
        let result = ceil_builtin(Value::Num(2.34), args).expect("ceil");
        match result {
            Value::Num(v) => assert!((v - 2.4).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_nan_and_inf_preserved() {
        let tensor =
            Tensor::new(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY], vec![3, 1]).unwrap();
        let result = ceil_builtin(Value::Tensor(tensor), Vec::new()).expect("ceil");
        match result {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert!(t.data[1].is_infinite() && t.data[1].is_sign_positive());
                assert!(t.data[2].is_infinite() && t.data[2].is_sign_negative());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_string_input_errors() {
        let err = ceil_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert_error_contains(err, "numeric");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_like_invalid_prototype_errors() {
        let args = vec![Value::from("like"), Value::from("prototype")];
        let err = ceil_builtin(Value::Num(1.0), args).unwrap_err();
        assert_error_contains(err, "unsupported prototype");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_like_missing_prototype_errors() {
        let err = ceil_builtin(Value::Num(1.0), vec![Value::from("like")]).unwrap_err();
        assert_error_contains(err, "expected prototype");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_like_host_output_keeps_host_residency() {
        let args = vec![Value::from("like"), Value::Num(0.0)];
        let result = ceil_builtin(Value::Num(1.2), args).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, 2.0),
            other => panic!("expected host scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_like_gpu_output() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.9, -1.2, 2.7, -3.4], vec![2, 2]).unwrap();
            let like_proto = {
                let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
                let view = HostTensorView {
                    data: &proto.data,
                    shape: &proto.shape,
                };
                provider.upload(&view).expect("upload proto")
            };
            let args = vec![Value::from("like"), Value::GpuTensor(like_proto)];
            let result = ceil_builtin(Value::Tensor(tensor), args).expect("ceil");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    assert_eq!(gathered.data, vec![1.0, -1.0, 3.0, -3.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_decimal_digits_with_gpu_like_prototype_reuploads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.901, -1.216], vec![2, 1]).unwrap();
            let tensor_view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let input_handle = provider.upload(&tensor_view).expect("upload input");

            let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto_tensor.data,
                shape: &proto_tensor.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let args = vec![
                Value::Int(IntValue::I32(2)),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ];
            let result = ceil_builtin(Value::GpuTensor(input_handle), args).expect("ceil");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.data, vec![0.91, -1.21]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ceil_bool_value() {
        let result = ceil_builtin(Value::Bool(true), Vec::new()).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ceil_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.3, 1.1, -0.2, -1.7], vec![2, 2]).unwrap();
        let cpu = ceil_numeric(Value::Tensor(t.clone()), CeilStrategy::Integer).unwrap();
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(ceil_gpu(
            h,
            &CeilArgs {
                strategy: CeilStrategy::Integer,
                output: OutputTemplate::Default,
            },
        ))
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                assert_eq!(gt.data, ct.data);
            }
            (Value::Num(c), gt) => {
                assert_eq!(gt.data, vec![c]);
            }
            other => panic!("unexpected comparison {other:?}"),
        }
    }
}
