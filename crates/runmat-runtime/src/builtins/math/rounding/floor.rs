//! MATLAB-compatible `floor` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::floor")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "floor",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_floor" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute floor directly on the device; the runtime gathers to the host when unary_floor is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::floor")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "floor",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("floor({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `floor` calls; providers can substitute custom kernels when available.",
};

const BUILTIN_NAME: &str = "floor";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "floor",
    category = "math/rounding",
    summary = "Round values toward negative infinity.",
    keywords = "floor,rounding,integers,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::rounding::floor"
)]
async fn floor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_arguments(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => floor_gpu(handle, &args).await?,
        Value::Complex(re, im) => Value::Complex(
            apply_floor_scalar(re, args.strategy),
            apply_floor_scalar(im, args.strategy),
        ),
        Value::ComplexTensor(ct) => floor_complex_tensor(ct, args.strategy)?,
        Value::CharArray(ca) => floor_char_array(ca, args.strategy)?,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            let floored = floor_tensor(tensor, args.strategy)?;
            tensor::tensor_into_value(floored)
        }
        Value::String(_) | Value::StringArray(_) => {
            return Err(builtin_error("floor: expected numeric or logical input"));
        }
        other => floor_numeric(other, args.strategy)?,
    };
    apply_output_template(base, &args.output).await
}

fn floor_numeric(value: Value, strategy: FloorStrategy) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("floor", value).map_err(|err| builtin_error(err))?;
    let floored = floor_tensor(tensor, strategy)?;
    Ok(tensor::tensor_into_value(floored))
}

fn floor_tensor(mut tensor: Tensor, strategy: FloorStrategy) -> BuiltinResult<Tensor> {
    for value in &mut tensor.data {
        *value = apply_floor_scalar(*value, strategy);
    }
    Ok(tensor)
}

fn floor_complex_tensor(ct: ComplexTensor, strategy: FloorStrategy) -> BuiltinResult<Value> {
    let data: Vec<(f64, f64)> = ct
        .data
        .iter()
        .map(|&(re, im)| {
            (
                apply_floor_scalar(re, strategy),
                apply_floor_scalar(im, strategy),
            )
        })
        .collect();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error(format!("floor: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn floor_char_array(ca: CharArray, strategy: FloorStrategy) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(apply_floor_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("floor: {e}")))?;
    Ok(Value::Tensor(tensor))
}

async fn floor_gpu(handle: GpuTensorHandle, args: &FloorArgs) -> BuiltinResult<Value> {
    if matches!(args.strategy, FloorStrategy::Integer) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_floor(&handle).await {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let floored = floor_tensor(tensor, args.strategy)?;
    Ok(tensor::tensor_into_value(floored))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloorStrategy {
    Integer,
    Decimals(i32),
    Significant(i32),
}

#[derive(Clone, Debug)]
struct FloorArgs {
    strategy: FloorStrategy,
    output: OutputTemplate,
}

#[derive(Clone, Debug)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<FloorArgs> {
    let (strategy_len, output) = parse_output_template(args)?;
    let strategy = match strategy_len {
        0 => FloorStrategy::Integer,
        1 => FloorStrategy::Decimals(parse_digits(&args[0])?),
        2 => {
            let digits = parse_digits(&args[0])?;
            let mode = parse_mode(&args[1])?;
            match mode {
                FloorMode::Decimals => FloorStrategy::Decimals(digits),
                FloorMode::Significant => {
                    if digits <= 0 {
                        return Err(builtin_error(
                            "floor: N must be a positive integer for 'significant' rounding",
                        ));
                    }
                    FloorStrategy::Significant(digits)
                }
            }
        }
        _ => return Err(builtin_error("floor: too many input arguments")),
    };
    Ok(FloorArgs { strategy, output })
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<(usize, OutputTemplate)> {
    if !args.is_empty() && is_keyword(&args[args.len() - 1], "like") {
        return Err(builtin_error("floor: expected prototype after 'like'"));
    }
    if args.len() >= 2 && is_keyword(&args[args.len() - 2], "like") {
        let proto = &args[args.len() - 1];
        if matches!(
            proto,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err(builtin_error("floor: unsupported prototype for 'like'"));
        }
        return Ok((args.len() - 2, OutputTemplate::Like(proto.clone())));
    }
    Ok((args.len(), OutputTemplate::Default))
}

fn parse_digits(value: &Value) -> BuiltinResult<i32> {
    let err = || builtin_error("floor: N must be an integer scalar");
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
            return Err(builtin_error(format!(
                "floor: N must be numeric, got {:?}",
                other
            )))
        }
    };
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err(builtin_error("floor: integer overflow in N"));
    }
    Ok(raw as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloorMode {
    Decimals,
    Significant,
}

fn parse_mode(value: &Value) -> BuiltinResult<FloorMode> {
    let Some(text) = tensor::value_to_string(value) else {
        return Err(builtin_error(
            "floor: mode must be a character vector or string scalar",
        ));
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(FloorMode::Significant),
        "decimal" | "decimals" => Ok(FloorMode::Decimals),
        other => Err(builtin_error(format!(
            "floor: unknown rounding mode '{other}'"
        ))),
    }
}

fn is_keyword(value: &Value, target: &str) -> bool {
    tensor::value_to_string(value)
        .map(|s| s.trim().eq_ignore_ascii_case(target))
        .unwrap_or(false)
}

fn apply_floor_scalar(value: f64, strategy: FloorStrategy) -> f64 {
    if !value.is_finite() {
        return value;
    }
    match strategy {
        FloorStrategy::Integer => value.floor(),
        FloorStrategy::Decimals(digits) => floor_with_decimals(value, digits),
        FloorStrategy::Significant(digits) => floor_with_significant(value, digits),
    }
}

fn floor_with_decimals(value: f64, digits: i32) -> f64 {
    if digits == 0 {
        return value.floor();
    }
    let factor = 10f64.powi(digits);
    if !factor.is_finite() || factor == 0.0 {
        return value;
    }
    (value * factor).floor() / factor
}

fn floor_with_significant(value: f64, digits: i32) -> f64 {
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
    (value * scale).floor() / scale
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
                "floor: unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        builtin_error(
            "floor: GPU output requested via 'like' but no acceleration provider is active",
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
                .map_err(|e| builtin_error(format!("floor: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("floor: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        other => Err(builtin_error(format!(
            "floor: 'like' GPU prototypes are only supported for real numeric outputs (got {other:?})"
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
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Tensor, Type, Value};

    fn floor_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::floor_builtin(value, rest))
    }

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[test]
    fn floor_type_preserves_tensor_shape() {
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
    fn floor_type_scalar_tensor_returns_num() {
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
    fn floor_scalar_positive_and_negative() {
        let value = Value::Num(-2.7);
        let result = floor_builtin(value, Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, -3.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_integer_tensor() {
        let tensor = Tensor::new(vec![1.2, 4.7, -3.4, 5.0], vec![2, 2]).unwrap();
        let result = floor_builtin(Value::Tensor(tensor), Vec::new()).expect("floor");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 4.0, -4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_complex_value() {
        let result = floor_builtin(Value::Complex(1.7, -2.3), Vec::new()).expect("floor");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -3.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_char_array_to_tensor() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = floor_builtin(Value::CharArray(chars), Vec::new()).expect("floor");
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
    fn floor_logical_array_remains_same() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = floor_builtin(Value::LogicalArray(logical), Vec::new()).expect("floor");
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
    fn floor_int_value_passthrough() {
        let result = floor_builtin(Value::Int(IntValue::I32(-4)), Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, -4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.2, 1.9, -0.1, -3.8], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = floor_builtin(Value::GpuTensor(handle), Vec::new()).expect("floor");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![0.0, 1.0, -1.0, -4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_decimal_digits() {
        let value = Value::Num(21.456);
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = floor_builtin(value, args).expect("floor");
        match result {
            Value::Num(v) => assert!((v - 21.45).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_negative_digits() {
        let tensor = Tensor::new(vec![123.4, -987.6], vec![2, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-2))];
        let result = floor_builtin(Value::Tensor(tensor), args).expect("floor");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![100.0, -1000.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_significant_digits() {
        let value = Value::Num(98765.4321);
        let args = vec![Value::Int(IntValue::I32(3)), Value::from("significant")];
        let result = floor_builtin(value, args).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, 98700.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_significant_requires_positive_digits() {
        let args = vec![Value::Int(IntValue::I32(0)), Value::from("significant")];
        let err = floor_builtin(Value::Num(1.23), args).unwrap_err();
        assert_error_contains(err, "positive integer");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_string_input_errors() {
        let err = floor_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert_error_contains(err, "numeric");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_like_invalid_prototype_errors() {
        let args = vec![Value::from("like"), Value::from("prototype")];
        let err = floor_builtin(Value::Num(1.0), args).unwrap_err();
        assert_error_contains(err, "unsupported prototype");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_like_gpu_output() {
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
            let result = floor_builtin(Value::Tensor(tensor), args).expect("floor");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    assert_eq!(gathered.data, vec![0.0, -2.0, 2.0, -4.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn floor_bool_value() {
        let result = floor_builtin(Value::Bool(true), Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn floor_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.3, 1.1, -0.2, -1.7], vec![2, 2]).unwrap();
        let cpu = floor_numeric(Value::Tensor(t.clone()), FloorStrategy::Integer).unwrap();
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(floor_gpu(
            h,
            &FloorArgs {
                strategy: FloorStrategy::Integer,
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
