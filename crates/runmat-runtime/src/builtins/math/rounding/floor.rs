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
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "floor",
        wasm_path = "crate::builtins::math::rounding::floor"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "floor"
category: "math/rounding"
keywords: ["floor", "rounding", "digits", "significant digits", "gpu", "like"]
summary: "Round scalars, vectors, matrices, or N-D tensors toward negative infinity or to specified digits."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_floor."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::floor::tests"
  integration: "builtins::math::rounding::floor::tests::floor_gpu_provider_roundtrip"
---

# What does the `floor` function do in MATLAB / RunMat?
`floor(X)` rounds each element of `X` toward negative infinity, returning the greatest integer less than or equal to the input. Optional arguments match MATLAB's extended syntax for rounding to fixed decimal places, significant digits, and prototype-based residency.

## How does the `floor` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and higher-dimensional tensors with MATLAB broadcasting semantics.
- `floor(X, N)` rounds toward negative infinity with `N` decimal digits (positive `N`) or powers of ten (negative `N`).
- `floor(X, N, 'significant')` rounds to `N` significant digits; `N` must be a positive integer.
- Logical inputs are promoted to doubles (`false → 0`, `true → 1`) before flooring.
- Character arrays are interpreted numerically (their Unicode code points) and return dense double tensors.
- Complex inputs are floored component-wise: `floor(a + bi) = floor(a) + i·floor(b)`.
- Non-finite values (`NaN`, `Inf`, `-Inf`) propagate unchanged.
- Empty arrays return empty arrays of the appropriate shape.
- Appending `'like', prototype` forces the result to match the residency of `prototype` (CPU or GPU). Currently prototypes must be numeric.

## `floor` Function GPU Execution Behaviour
When tensors already reside on the GPU, RunMat consults the active acceleration provider. If the provider implements the `unary_floor` hook, `floor(X)` executes entirely on the device and keeps tensors resident. When decimal or significant-digit rounding is requested—or when the provider lacks `unary_floor`—RunMat gathers the tensor to host memory, applies the CPU implementation, and honours any `'like'` GPU prototype by uploading the result back to the device. This keeps semantics consistent even when specialised kernels are unavailable.

## Examples of using the `floor` function in MATLAB / RunMat

### Flooring positive and negative scalars

```matlab
x = [-2.7, -0.3, 0, 0.8, 3.9];
y = floor(x);
```

Expected output:

```matlab
y = [-3, -1, 0, 0, 3];
```

### Flooring every element of a matrix

```matlab
A = [1.2 4.7; -3.4 5.0];
B = floor(A);
```

Expected output:

```matlab
B = [1 4; -4 5];
```

### Flooring fractions stored in a tensor

```matlab
t = reshape([-1.8, -0.2, 0.4, 1.9, 2.1, 3.6], [3, 2]);
floored = floor(t);
```

Expected output:

```matlab
floored =
    [-2  1;
     -1  2;
      0  3]
```

### Flooring values to a fixed number of decimal places

```matlab
temps = [21.456 19.995 22.501];
floored = floor(temps, 2);
```

Expected output:

```matlab
floored = [21.45 19.99 22.50];
```

### Flooring to significant digits

```matlab
measurements = [0.001234 12.3456 98765];
sig2 = floor(measurements, 2, 'significant');
```

Expected output:

```matlab
sig2 = [0.0012 12.0 98000];
```

### Flooring complex numbers component-wise

```matlab
z = [1.7 + 2.1i, -0.2 - 3.9i];
result = floor(z);
```

Expected output:

```matlab
result = [1 + 2i, -1 - 4i];
```

### Keeping GPU data on device when the provider supports `unary_floor`

```matlab
G = gpuArray([1.8 -0.2 0.0; -1.1 2.5 -3.4]);
floored = floor(G);
H = gather(floored);
```

Expected output:

```matlab
H =
    [ 1 -1  0;
     -2  2 -4]
```

### Forcing GPU residency with a `'like'` prototype

```matlab
A = [1.8 -0.2; 2.7 3.4];
proto = gpuArray(0);
G = floor(A, 'like', proto);   % Result remains on the GPU
result = gather(G);
```

Expected output:

```matlab
result =
    [ 1 -1;
      2  3]
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. RunMat's planner keeps tensors on the GPU when a provider implements `unary_floor` and the workload benefits from device execution. When the hook is missing, RunMat automatically gathers the data, applies the CPU semantics, and allows subsequent operations to re-upload if profitable. Explicit `gpuArray` calls remain available for compatibility with MathWorks MATLAB.

## FAQ

1. **Does `floor` always round toward negative infinity?** Yes—positive values round down toward zero, while negative values round to the more negative integer (e.g., `floor(-0.1) = -1`).
2. **How are complex numbers handled?** The real and imaginary parts are floored independently, matching MATLAB's component-wise definition.
3. **Can I round to decimal digits or significant digits?** Yes. Use `floor(X, N)` for decimal digits or `floor(X, N, 'significant')` for significant digits. Negative `N` values round to powers of ten.
4. **What happens with logical arrays?** Logical values promote to doubles (`0` or `1`) before flooring, so the outputs remain 0 or 1.
5. **Can I pass character arrays to `floor`?** Yes. Character data is treated as its numeric code points, producing a double tensor of the same size.
6. **Do `NaN` and `Inf` values change?** No. Non-finite inputs propagate unchanged.
7. **Will GPU execution change floating-point results?** No. Providers implement IEEE-compliant flooring; when a provider lacks `unary_floor`, RunMat falls back to the CPU to preserve MATLAB-compatible behaviour.
8. **Does `'like'` work with `floor`?** Yes. Append `'like', prototype` to request output that matches the prototype's residency. Currently prototypes must be numeric (scalars or dense tensors, host or GPU).
9. **Can fusion keep `floor` on the GPU?** Yes. `floor` participates in elementwise fusion, so fused graphs can stay resident on the device when supported.

## See Also
[ceil](./ceil), [round](./round), [fix](./fix), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for `floor` lives at: [`crates/runmat-runtime/src/builtins/math/rounding/floor.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/floor.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::rounding::floor")]
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

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::rounding::floor")]
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

#[runtime_builtin(
    name = "floor",
    category = "math/rounding",
    summary = "Round values toward negative infinity.",
    keywords = "floor,rounding,integers,gpu",
    accel = "unary",
    wasm_path = "crate::builtins::math::rounding::floor"
)]
fn floor_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let args = parse_arguments(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => floor_gpu(handle, &args)?,
        Value::Complex(re, im) => Value::Complex(
            apply_floor_scalar(re, args.strategy),
            apply_floor_scalar(im, args.strategy),
        ),
        Value::ComplexTensor(ct) => floor_complex_tensor(ct, args.strategy)?,
        Value::CharArray(ca) => floor_char_array(ca, args.strategy)?,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            let floored = floor_tensor(tensor, args.strategy)?;
            tensor::tensor_into_value(floored)
        }
        Value::String(_) | Value::StringArray(_) => {
            return Err("floor: expected numeric or logical input".to_string())
        }
        other => floor_numeric(other, args.strategy)?,
    };
    apply_output_template(base, &args.output)
}

fn floor_numeric(value: Value, strategy: FloorStrategy) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("floor", value)?;
    let floored = floor_tensor(tensor, strategy)?;
    Ok(tensor::tensor_into_value(floored))
}

fn floor_tensor(mut tensor: Tensor, strategy: FloorStrategy) -> Result<Tensor, String> {
    for value in &mut tensor.data {
        *value = apply_floor_scalar(*value, strategy);
    }
    Ok(tensor)
}

fn floor_complex_tensor(ct: ComplexTensor, strategy: FloorStrategy) -> Result<Value, String> {
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
    let tensor = ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("floor: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn floor_char_array(ca: CharArray, strategy: FloorStrategy) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(apply_floor_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("floor: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn floor_gpu(handle: GpuTensorHandle, args: &FloorArgs) -> Result<Value, String> {
    if matches!(args.strategy, FloorStrategy::Integer) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_floor(&handle) {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
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

fn parse_arguments(args: &[Value]) -> Result<FloorArgs, String> {
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
                        return Err(
                            "floor: N must be a positive integer for 'significant' rounding"
                                .to_string(),
                        );
                    }
                    FloorStrategy::Significant(digits)
                }
            }
        }
        _ => return Err("floor: too many input arguments".to_string()),
    };
    Ok(FloorArgs { strategy, output })
}

fn parse_output_template(args: &[Value]) -> Result<(usize, OutputTemplate), String> {
    if !args.is_empty() && is_keyword(&args[args.len() - 1], "like") {
        return Err("floor: expected prototype after 'like'".to_string());
    }
    if args.len() >= 2 && is_keyword(&args[args.len() - 2], "like") {
        let proto = &args[args.len() - 1];
        if matches!(
            proto,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err("floor: unsupported prototype for 'like'".to_string());
        }
        return Ok((args.len() - 2, OutputTemplate::Like(proto.clone())));
    }
    Ok((args.len(), OutputTemplate::Default))
}

fn parse_digits(value: &Value) -> Result<i32, String> {
    let err = || "floor: N must be an integer scalar".to_string();
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
        other => return Err(format!("floor: N must be numeric, got {:?}", other)),
    };
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err("floor: integer overflow in N".to_string());
    }
    Ok(raw as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloorMode {
    Decimals,
    Significant,
}

fn parse_mode(value: &Value) -> Result<FloorMode, String> {
    let Some(text) = tensor::value_to_string(value) else {
        return Err("floor: mode must be a character vector or string scalar".to_string());
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(FloorMode::Significant),
        "decimal" | "decimals" => Ok(FloorMode::Decimals),
        other => Err(format!("floor: unknown rounding mode '{other}'")),
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

fn apply_output_template(value: Value, output: &OutputTemplate) -> Result<Value, String> {
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
            | Value::ComplexTensor(_) => convert_to_host_like(value),
            _ => Err(
                "floor: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "floor: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("floor: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("floor: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        other => Err(format!(
            "floor: 'like' GPU prototypes are only supported for real numeric outputs (got {other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("floor: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray, Tensor, Value};

    #[test]
    fn floor_scalar_positive_and_negative() {
        let value = Value::Num(-2.7);
        let result = floor_builtin(value, Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, -3.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

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

    #[test]
    fn floor_int_value_passthrough() {
        let result = floor_builtin(Value::Int(IntValue::I32(-4)), Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, -4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

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

    #[test]
    fn floor_significant_requires_positive_digits() {
        let args = vec![Value::Int(IntValue::I32(0)), Value::from("significant")];
        let err = floor_builtin(Value::Num(1.23), args).unwrap_err();
        assert!(err.contains("positive integer"), "unexpected error: {err}");
    }

    #[test]
    fn floor_string_input_errors() {
        let err = floor_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert!(err.contains("numeric"), "unexpected error: {err}");
    }

    #[test]
    fn floor_like_invalid_prototype_errors() {
        let args = vec![Value::from("like"), Value::from("prototype")];
        let err = floor_builtin(Value::Num(1.0), args).unwrap_err();
        assert!(err.contains("unsupported prototype"), "unexpected: {err}");
    }

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

    #[test]
    fn floor_bool_value() {
        let result = floor_builtin(Value::Bool(true), Vec::new()).expect("floor");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

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
        let gpu = floor_gpu(
            h,
            &FloorArgs {
                strategy: FloorStrategy::Integer,
                output: OutputTemplate::Default,
            },
        )
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
