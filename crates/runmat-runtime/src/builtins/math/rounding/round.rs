//! MATLAB-compatible `round` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "round",
        builtin_path = "crate::builtins::math::rounding::round"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "round"
category: "math/rounding"
keywords: ["round", "rounding", "significant digits", "decimals", "gpu"]
summary: "Round scalars, vectors, matrices, or N-D tensors to the nearest integers, decimal digits, or significant digits."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when rounding to digits or significant figures if the active provider lacks unary_round."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::round::tests"
  integration: "builtins::math::rounding::round::tests::round_gpu_provider_roundtrip"
---

# What does the `round` function do in MATLAB / RunMat?
`round(X)` rounds numeric values to the nearest integers using MATLAB-compatible half-away-from-zero semantics. Additional arguments allow rounding to a specified number of decimal digits or significant digits.

## How does the `round` function behave in MATLAB / RunMat?
- `round(X)` rounds each element of `X` to the nearest integer; ties (e.g., `±0.5`) round away from zero.
- `round(X, N)` rounds to `N` decimal digits when `N` is positive and to powers of ten when `N` is negative.
- `round(X, N, 'significant')` rounds to `N` significant digits. `N` must be a positive integer.
- Logical inputs are promoted to double before rounding; `round(true)` returns `1`.
- Complex inputs are rounded component-wise (`round(a + bi) = round(a) + i·round(b)`), matching MATLAB.
- Non-finite values (`NaN`, `Inf`, `-Inf`) propagate unchanged regardless of precision arguments.
- Character arrays are treated as their numeric code points and return double tensors of the same size.

## `round` Function GPU Execution Behaviour
When a tensor already resides on the GPU, RunMat checks whether the active acceleration provider implements a specialised `unary_round` kernel. If available, `round(X)` executes entirely on the device. Advanced modes (`round(X, N)` and `round(X, N, 'significant')`) currently gather tensors to the host before rounding to keep semantics aligned with MATLAB. Providers that add digit-aware kernels can extend this path in the future.

## Examples of using the `round` function in MATLAB / RunMat

### Rounding values to the nearest integers

```matlab
X = [-3.5 -2.2 -0.5 0 0.5 1.7];
Y = round(X);
```

Expected output:

```matlab
Y = [-4 -2 -1 0 1 2];
```

### Rounding to a fixed number of decimal places

```matlab
temps = [21.456 19.995 22.501];
rounded = round(temps, 2);
```

Expected output:

```matlab
rounded = [21.46 20.00 22.50];
```

### Rounding to negative powers of ten

```matlab
counts = [1234 5678 91011];
rounded = round(counts, -2);
```

Expected output:

```matlab
rounded = [1200 5700 91000];
```

### Rounding to significant digits

```matlab
measurements = [0.001234 12.3456 98765];
sig3 = round(measurements, 3, 'significant');
```

Expected output:

```matlab
sig3 = [0.00123 12.3 98800];
```

### Rounding GPU tensors and gathering the results

```matlab
G = gpuArray(linspace(-2.5, 2.5, 6));
rounded = round(G);
hostValues = gather(rounded);
```

Expected output:

```matlab
hostValues = [-3 -2 -1 1 2 3];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. RunMat's planner keeps tensors on the GPU when the provider exposes the required kernels and it is profitable to do so. `round` takes advantage of this mechanism for the plain `round(X)` form. When you specify digits or the `'significant'` option, RunMat currently gathers data to the host to match MATLAB exactly. Future providers can extend `unary_round` or add digit-aware kernels to keep those workloads on the device.

## FAQ

### Does `round` always round half values away from zero?
Yes. MATLAB and RunMat both use half-away-from-zero semantics, so `round(0.5)` returns `1` and `round(-0.5)` returns `-1`.

### Can I round to decimal places and significant digits?
Yes. Use `round(X, N)` for decimal places and `round(X, N, 'significant')` for significant digits. Negative `N` values round to tens, hundreds, and so on.

### What happens if I pass a non-integer `N`?
`N` must be an integer scalar. RunMat raises a MATLAB-compatible error when `N` is not an integer or is non-finite.

### How are complex numbers handled?
RunMat rounds the real and imaginary components independently, matching MATLAB's component-wise behaviour.

### Do NaN or Inf values change when rounded?
No. Non-finite values propagate unchanged for every rounding mode, just like MATLAB.

### Will rounding stay on the GPU?
`round(X)` stays on the GPU when the provider implements `unary_round`. Rounding with digit arguments currently gathers to the host; providers can override this by adding specialised kernels.

### Can I round logical or character arrays?
Yes. Logical values are converted to doubles (`0` or `1`) and characters are rounded as their numeric code points, returning dense double tensors.

## See Also
[floor](./floor), [ceil](./ceil), [fix](./fix), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for `round` lives at: [`crates/runmat-runtime/src/builtins/math/rounding/round.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/round.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

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

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
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
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            Ok(round_tensor(tensor, strategy).map(tensor::tensor_into_value)?)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("round: expected numeric or logical input"))
        }
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
            let tensor =
                tensor::value_into_tensor_for("round", other).map_err(|err| builtin_error(err))?;
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
        .map_err(|e| builtin_error(format!("round: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn round_char_array(ca: CharArray, strategy: RoundStrategy) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ca.data.len());
    for ch in ca.data {
        data.push(round_scalar(ch as u32 as f64, strategy));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("round: {e}")))?;
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
                        return Err(builtin_error(
                            "round: N must be a positive integer for 'significant' rounding",
                        ));
                    }
                    Ok(RoundStrategy::Significant(digits))
                }
            }
        }
        _ => Err(builtin_error("round: too many input arguments")),
    }
}

fn parse_digits(value: &Value) -> BuiltinResult<i32> {
    let err = || builtin_error("round: N must be an integer scalar");
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
                "round: N must be numeric, got {:?}",
                other
            )))
        }
    };
    if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
        return Err(builtin_error("round: integer overflow in N"));
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
        return Err(builtin_error(
            "round: mode must be a character vector or string scalar",
        ));
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "significant" => Ok(RoundMode::Significant),
        "decimal" | "decimals" => Ok(RoundMode::Decimals),
        other => Err(builtin_error(format!(
            "round: unknown rounding mode '{other}'"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor};

    fn round_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::round_builtin(value, rest))
    }

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "unexpected error: {}",
            err.message()
        );
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
        assert_error_contains(err, "unknown rounding mode");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
