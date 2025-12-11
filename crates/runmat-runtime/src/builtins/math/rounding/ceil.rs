//! MATLAB-compatible `ceil` builtin with GPU-aware semantics for RunMat.

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
        name = "ceil",
        wasm_path = "crate::builtins::math::rounding::ceil"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ceil"
category: "math/rounding"
keywords: ["ceil", "rounding", "digits", "significant digits", "gpu", "like"]
summary: "Round scalars, vectors, matrices, or N-D tensors toward positive infinity or to specified digits."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_ceil."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::ceil::tests"
  integration: "builtins::math::rounding::ceil::tests::ceil_gpu_provider_roundtrip"
---

# What does the `ceil` function do in MATLAB / RunMat?
`ceil(X)` rounds each element of `X` toward positive infinity, returning the smallest integer greater than or equal to the input. Optional arguments match MATLAB's extended syntax for rounding to fixed decimal places, significant digits, and prototype-based residency.

## How does the `ceil` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and higher-dimensional tensors with MATLAB broadcasting semantics.
- `ceil(X, N)` rounds toward positive infinity with `N` decimal digits (positive `N`) or powers of ten (negative `N`).
- `ceil(X, N, 'significant')` rounds to `N` significant digits; `N` must be a positive integer.
- Logical inputs are promoted to doubles (`false → 0`, `true → 1`) before applying the ceiling operation.
- Character arrays are interpreted numerically (their Unicode code points) and return dense double tensors.
- Complex inputs are rounded component-wise: `ceil(a + bi) = ceil(a) + i·ceil(b)`.
- Non-finite values (`NaN`, `Inf`, `-Inf`) propagate unchanged.
- Empty arrays return empty arrays of the appropriate shape.
- Appending `'like', prototype` forces the result to match the residency of `prototype` (CPU or GPU). Prototypes must currently be numeric.

## `ceil` Function GPU Execution Behaviour
When tensors already reside on the GPU, RunMat consults the active acceleration provider. If the provider implements the `unary_ceil` hook, `ceil(X)` executes entirely on the device and keeps tensors resident. When decimal or significant-digit rounding is requested—or when the provider lacks `unary_ceil`—RunMat gathers the tensor to host memory, applies the CPU implementation, and honours any `'like'` GPU prototype by uploading the result back to the device. This maintains MATLAB-compatible behaviour while exposing GPU acceleration whenever it is available.

## Examples of using the `ceil` function in MATLAB / RunMat

### Rounding values up to the next integer

```matlab
x = [-2.7, -0.3, 0, 0.8, 3.2];
y = ceil(x);
```

Expected output:

```matlab
y = [-2, 0, 0, 1, 4];
```

### Rounding a matrix up element-wise

```matlab
A = [1.2 4.7; -3.4 5.0];
B = ceil(A);
```

Expected output:

```matlab
B = [2 5; -3 5];
```

### Rounding fractions upward in a tensor

```matlab
t = reshape([-1.8, -0.2, 0.4, 1.1, 2.1, 3.6], [3, 2]);
up = ceil(t);
```

Expected output:

```matlab
up =
    [-1  2;
      0  3;
      1  4]
```

### Rounding up to two decimal places

```matlab
temps = [21.452 19.991 22.501];
rounded = ceil(temps, 2);
```

Expected output:

```matlab
rounded = [21.46 19.99 22.51];
```

### Rounding to significant digits

```matlab
measurements = [0.001234 12.3456 98765];
sig2 = ceil(measurements, 2, 'significant');
```

Expected output:

```matlab
sig2 = [0.0013 13.0 99000];
```

### Rounding complex numbers toward positive infinity

```matlab
z = [1.2 + 2.1i, -0.2 - 3.9i];
result = ceil(z);
```

Expected output:

```matlab
result = [2 + 3i, 0 - 3i];
```

### Keeping GPU results on-device with `unary_ceil`

```matlab
G = gpuArray([1.8 -0.2 0.0; -1.1 2.5 -3.4]);
up = ceil(G);
gather(up);
```

Expected output:

```matlab
ans =
    [ 2  0  0;
     -1  3 -3]
```

### Forcing GPU residency with a `'like'` prototype

```matlab
A = [1.8 -0.2; 2.7 3.4];
proto = gpuArray(0);
G = ceil(A, 'like', proto);   % Result remains on the GPU
result = gather(G);
```

Expected output:

```matlab
result =
    [ 2  0;
      3  4]
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. RunMat's planner keeps tensors on the GPU when a provider implements `unary_ceil` and the workload benefits from device execution. When the hook is missing, RunMat automatically gathers the data, applies the CPU semantics, and allows subsequent operations to re-upload if profitable. Explicit `gpuArray` calls remain available for compatibility with MathWorks MATLAB.

## FAQ

1. **Does `ceil` always round toward positive infinity?** Yes—positive values round up away from zero, while negative values round toward zero (e.g., `ceil(-0.1) = 0`).
2. **How are complex numbers handled?** The real and imaginary parts are ceiled independently, matching MATLAB's component-wise definition.
3. **Can I round to decimal digits or significant digits?** Yes. Use `ceil(X, N)` for decimal digits or `ceil(X, N, 'significant')` for significant digits. Negative `N` values round to powers of ten.
4. **What happens with logical arrays?** Logical values promote to doubles (`0` or `1`) before rounding, so the outputs remain 0 or 1.
5. **Can I pass character arrays to `ceil`?** Yes. Character data is treated as its numeric code points, producing a double tensor of the same size.
6. **Do `NaN` and `Inf` values change?** No. Non-finite inputs propagate unchanged.
7. **Will GPU execution change floating-point results?** No. Providers implement IEEE-compliant ceiling; when a provider lacks `unary_ceil`, RunMat falls back to the CPU to preserve MATLAB-compatible behaviour.
8. **Does `'like'` work with `ceil`?** Yes. Append `'like', prototype` to request output that matches the prototype's residency. Currently prototypes must be numeric (scalars or dense tensors, host or GPU).
9. **Can fusion keep `ceil` on the GPU?** Yes. `ceil` participates in elementwise fusion, so fused graphs can stay resident on the device when supported.

## See Also
[floor](./floor), [round](./round), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for `ceil` lives at: [`crates/runmat-runtime/src/builtins/math/rounding/ceil.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/ceil.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::rounding::ceil")]
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

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::rounding::ceil")]
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

#[runtime_builtin(
    name = "ceil",
    category = "math/rounding",
    summary = "Round values toward positive infinity.",
    keywords = "ceil,rounding,integers,gpu",
    accel = "unary",
    wasm_path = "crate::builtins::math::rounding::ceil"
)]
fn ceil_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let args = parse_arguments(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => ceil_gpu(handle, &args)?,
        Value::Complex(re, im) => Value::Complex(
            apply_ceil_scalar(re, args.strategy),
            apply_ceil_scalar(im, args.strategy),
        ),
        Value::ComplexTensor(ct) => ceil_complex_tensor(ct, args.strategy)?,
        Value::CharArray(ca) => ceil_char_array(ca, args.strategy)?,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            let ceiled = ceil_tensor(tensor, args.strategy)?;
            tensor::tensor_into_value(ceiled)
        }
        Value::String(_) | Value::StringArray(_) => {
            return Err("ceil: expected numeric or logical input".to_string())
        }
        other => ceil_numeric(other, args.strategy)?,
    };
    apply_output_template(base, &args.output)
}

fn ceil_numeric(value: Value, strategy: CeilStrategy) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("ceil", value)?;
    let ceiled = ceil_tensor(tensor, strategy)?;
    Ok(tensor::tensor_into_value(ceiled))
}

fn ceil_tensor(mut tensor: Tensor, strategy: CeilStrategy) -> Result<Tensor, String> {
    for value in &mut tensor.data {
        *value = apply_ceil_scalar(*value, strategy);
    }
    Ok(tensor)
}

fn ceil_complex_tensor(ct: ComplexTensor, strategy: CeilStrategy) -> Result<Value, String> {
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
    let tensor = ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("ceil: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn ceil_char_array(ca: CharArray, strategy: CeilStrategy) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(apply_ceil_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("ceil: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn ceil_gpu(handle: GpuTensorHandle, args: &CeilArgs) -> Result<Value, String> {
    if matches!(args.strategy, CeilStrategy::Integer) {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
            if let Ok(out) = provider.unary_ceil(&handle) {
                return Ok(Value::GpuTensor(out));
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
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

fn parse_arguments(args: &[Value]) -> Result<CeilArgs, String> {
    let (strategy_len, output) = parse_output_template(args)?;
    let strategy = match strategy_len {
        0 => CeilStrategy::Integer,
        1 => CeilStrategy::Decimals(parse_digits(&args[0])?),
        2 => {
            let digits = parse_digits(&args[0])?;
            let mode = parse_mode(&args[1])?;
            match mode {
                CeilMode::Decimals => CeilStrategy::Decimals(digits),
                CeilMode::Significant => {
                    if digits <= 0 {
                        return Err(
                            "ceil: N must be a positive integer for 'significant' rounding"
                                .to_string(),
                        );
                    }
                    CeilStrategy::Significant(digits)
                }
            }
        }
        _ => return Err("ceil: too many input arguments".to_string()),
    };
    Ok(CeilArgs { strategy, output })
}

fn parse_output_template(args: &[Value]) -> Result<(usize, OutputTemplate), String> {
    if !args.is_empty() && is_keyword(&args[args.len() - 1], "like") {
        return Err("ceil: expected prototype after 'like'".to_string());
    }
    if args.len() >= 2 && is_keyword(&args[args.len() - 2], "like") {
        let proto = &args[args.len() - 1];
        if matches!(
            proto,
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
        ) {
            return Err("ceil: unsupported prototype for 'like'".to_string());
        }
        return Ok((args.len() - 2, OutputTemplate::Like(proto.clone())));
    }
    Ok((args.len(), OutputTemplate::Default))
}

fn parse_digits(value: &Value) -> Result<i32, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle.clone());
            let gathered = gpu_helpers::gather_value(&proxy).map_err(|e| format!("ceil: {e}"))?;
            parse_digits_inner(&gathered)
        }
        other => parse_digits_inner(other),
    }
}

fn parse_digits_inner(value: &Value) -> Result<i32, String> {
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
                return Err(ERR.to_string());
            }
            return digits_from_f64(tensor.data[0]);
        }
        Value::LogicalArray(logical) => {
            if logical.len() != 1 {
                return Err(ERR.to_string());
            }
            if logical.data[0] != 0 {
                1
            } else {
                0
            }
        }
        other => return Err(format!("ceil: N must be numeric, got {:?}", other)),
    };
    digits_from_i64(raw)
}

fn digits_from_f64(value: f64) -> Result<i32, String> {
    if !value.is_finite() {
        return Err("ceil: N must be an integer scalar".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("ceil: N must be an integer scalar".to_string());
    }
    if rounded > i64::MAX as f64 || rounded < i64::MIN as f64 {
        return Err("ceil: integer overflow in N".to_string());
    }
    digits_from_i64(rounded as i64)
}

fn digits_from_i64(raw: i64) -> Result<i32, String> {
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err("ceil: integer overflow in N".to_string());
    }
    Ok(raw as i32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CeilMode {
    Decimals,
    Significant,
}

fn parse_mode(value: &Value) -> Result<CeilMode, String> {
    let Some(text) = tensor::value_to_string(value) else {
        return Err("ceil: mode must be a character vector or string scalar".to_string());
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(CeilMode::Significant),
        "decimal" | "decimals" | "digits" | "places" | "place" => Ok(CeilMode::Decimals),
        other => Err(format!("ceil: unknown rounding mode '{other}'")),
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
                "ceil: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "ceil: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("ceil: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("ceil: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        other => Err(format!(
            "ceil: 'like' GPU prototypes are only supported for real numeric outputs (got {other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("ceil: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Tensor, Value};

    #[test]
    fn ceil_scalar_positive_and_negative() {
        let value = Value::Num(-2.7);
        let result = ceil_builtin(value, Vec::new()).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, -2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

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

    #[test]
    fn ceil_int_value_passthrough() {
        let result = ceil_builtin(Value::Int(IntValue::I32(-4)), Vec::new()).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, -4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

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

    #[test]
    fn ceil_significant_requires_positive_digits() {
        let args = vec![Value::Int(IntValue::I32(0)), Value::from("significant")];
        let err = ceil_builtin(Value::Num(1.23), args).unwrap_err();
        assert!(err.contains("positive integer"), "unexpected error: {err}");
    }

    #[test]
    fn ceil_decimal_mode_alias_digits_keyword() {
        let args = vec![Value::Int(IntValue::I32(1)), Value::from("digits")];
        let result = ceil_builtin(Value::Num(2.34), args).expect("ceil");
        match result {
            Value::Num(v) => assert!((v - 2.4).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

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

    #[test]
    fn ceil_string_input_errors() {
        let err = ceil_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert!(err.contains("numeric"), "unexpected error: {err}");
    }

    #[test]
    fn ceil_like_invalid_prototype_errors() {
        let args = vec![Value::from("like"), Value::from("prototype")];
        let err = ceil_builtin(Value::Num(1.0), args).unwrap_err();
        assert!(err.contains("unsupported prototype"), "unexpected: {err}");
    }

    #[test]
    fn ceil_like_missing_prototype_errors() {
        let err = ceil_builtin(Value::Num(1.0), vec![Value::from("like")]).unwrap_err();
        assert!(
            err.contains("expected prototype"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn ceil_like_host_output_keeps_host_residency() {
        let args = vec![Value::from("like"), Value::Num(0.0)];
        let result = ceil_builtin(Value::Num(1.2), args).expect("ceil");
        match result {
            Value::Num(v) => assert_eq!(v, 2.0),
            other => panic!("expected host scalar, got {other:?}"),
        }
    }

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

    #[test]
    fn ceil_bool_value() {
        let result = ceil_builtin(Value::Bool(true), Vec::new()).expect("ceil");
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
        let gpu = ceil_gpu(
            h,
            &CeilArgs {
                strategy: CeilStrategy::Integer,
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
