//! MATLAB-compatible `colon` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const MIN_RATIO_TOL: f64 = f64::EPSILON * 8.0;
const MAX_RATIO_TOL: f64 = 1e-9;
const ZERO_IM_TOL: f64 = f64::EPSILON * 32.0;
const CHAR_TOL: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarOrigin {
    Numeric,
    Char,
}

#[derive(Clone, Copy)]
struct ParsedScalar {
    value: f64,
    prefer_gpu: bool,
    origin: ScalarOrigin,
}

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "colon")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "colon"
category: "array/creation"
keywords: ["colon", "sequence", "range", "step", "gpu"]
summary: "Generate MATLAB-style arithmetic progressions with optional step size."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Prefers provider `linspace` kernels when available; otherwise the runtime uploads the host-generated vector so residency stays on device."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::creation::colon::tests"
  integration: "builtins::array::creation::colon::tests::colon_gpu_roundtrip"
  wgpu: "builtins::array::creation::colon::tests::colon_wgpu_matches_cpu"
---

# What does the `colon` function do in MATLAB / RunMat?
`colon(start, stop)` and `colon(start, step, stop)` build row vectors that mirror the MATLAB colon
operator. The function returns an arithmetic progression that begins at `start`, advances by
`step` (default `+1` or `-1` depending on the bounds), and stops before exceeding `stop`.

## How does the `colon` function behave in MATLAB / RunMat?
- Inputs must be real scalars (numeric, logical, scalar tensors, or single-character arrays). Imaginary parts must be zero.
- `colon(start, stop)` picks an increment of `+1` when `stop ≥ start`, otherwise `-1`.
- `colon(start, step, stop)` uses the supplied increment. A zero increment raises an error.
- When the endpoints are character scalars, the result is a row vector of characters; otherwise a double-precision row vector. Empty progressions are `1×0`.
- `stop` is included only when it lies on the arithmetic progression; otherwise the sequence stops
  at the last admissible value before overshooting.
- Arguments may be `gpuArray` scalars; the result stays on the GPU when a provider is active.
- Floating-point tolerance follows MATLAB’s rules, so values that should land on `stop` are preserved even when rounding noise accumulates.

## Examples of using the `colon` function in MATLAB / RunMat

### Generating consecutive integers

```matlab
x = colon(1, 5);
```

Expected output:

```matlab
x = [1 2 3 4 5];
```

### Counting down without specifying a step

```matlab
y = colon(5, 1);
```

Expected output:

```matlab
y = [5 4 3 2 1];
```

### Using a custom increment

```matlab
z = colon(0, 0.5, 2);
```

Expected output:

```matlab
z = [0 0.5 1.0 1.5 2.0];
```

### Stopping before overshooting the end point

```matlab
vals = colon(0, 2, 5);
```

Expected output:

```matlab
vals = [0 2 4];
```

### Working with fractional radians

```matlab
theta = colon(-pi, pi/4, pi/2);
```

Expected output:

```matlab
theta = [-3.1416 -2.3562 -1.5708 -0.7854 0.0000 0.7854 1.5708];
```

### Keeping sequences on the GPU

```matlab
g = gpuArray(0);
h = colon(g, 0.25, 1);
result = gather(h);
```

Expected behaviour:

```matlab
result = [0 0.25 0.5 0.75 1.0];
```

### Building character ranges

```matlab
letters = colon('a', 'f');
odds    = colon('a', 2, 'g');
```

Expected output:

```matlab
letters = 'abcdef';
odds    = 'aceg';
```

## GPU residency in RunMat (Do I need `gpuArray`?)

RunMat automatically keeps the output on the GPU when any input scalar already resides there and
an acceleration provider is active. Providers that implement the `linspace` hook (such as the wgpu
backend) generate the progression entirely on device. Other providers still return a GPU tensor by
uploading the host-generated vector, so downstream kernels can fuse without an extra gather.

## FAQ

### What happens when `start == stop`?
The output is a single-element vector containing `start`. With two arguments the implicit step is
`+1`, so the result is `[start]`.

### Why is `stop` sometimes missing from the result?
`stop` is included only when it aligns with the arithmetic progression. For example,
`colon(0, 2, 5)` produces `[0 2 4]` because `6` would overshoot the upper bound.

### Can I use zero or complex increments?
No. The increment must be a finite, non-zero real scalar. Supplying `0` or a value with a non-zero
imaginary part raises an error.

### Can I generate character sequences?
Yes. When both `start` and `stop` are single-character arrays, `colon` returns a character row vector.
Step values can still be numeric (for example, `colon('a', 2, 'g')` produces `'aceg'`).

### Does the function accept logical inputs?
Yes. Logical scalars are promoted to doubles (`true → 1`, `false → 0`) before building the
sequence.

### How does this differ from `linspace`?
`linspace(start, stop, n)` lets you pick the number of points directly, whereas `colon` fixes the
increment (`start:step:stop`). When `stop` is not exactly reachable, `colon` stops short instead of
nudging the final value.

## See Also
[linspace](./linspace), [logspace](./logspace), [range](./range), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `colon` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/colon.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/colon.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "colon",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linspace")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to uploading the host-generated vector when provider linspace kernels are unavailable.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "colon",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink; it does not participate in fusion.",
};

#[runtime_builtin(
    name = "colon",
    category = "array/creation",
    summary = "Arithmetic progression that mirrors MATLAB's colon operator.",
    keywords = "colon,sequence,range,step,gpu",
    accel = "array_construct"
)]
fn colon_builtin(start: Value, step_or_end: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("colon: expected two or three input arguments".to_string());
    }

    let start_scalar = parse_real_scalar("colon", start)?;

    if rest.is_empty() {
        let stop_scalar = parse_real_scalar("colon", step_or_end)?;
        let step = default_step(start_scalar.value, stop_scalar.value);
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        build_sequence(
            start_scalar.value,
            step,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    } else {
        let step_scalar = parse_real_scalar("colon", step_or_end)?;
        if step_scalar.value == 0.0 {
            return Err("colon: increment must be nonzero".to_string());
        }
        let stop_scalar = parse_real_scalar("colon", rest[0].clone())?;
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || step_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        build_sequence(
            start_scalar.value,
            step_scalar.value,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    }
}

fn build_sequence(
    start: f64,
    step: f64,
    stop: f64,
    explicit_gpu: bool,
    char_mode: bool,
) -> Result<Value, String> {
    if !start.is_finite() || !step.is_finite() || !stop.is_finite() {
        return Err("colon: inputs must be finite numeric scalars".to_string());
    }
    if step == 0.0 {
        return Err("colon: increment must be nonzero".to_string());
    }

    let plan = plan_progression(start, step, stop)?;

    if char_mode {
        let data = materialize_progression(&plan, start, step);
        return build_char_sequence(data);
    }

    if plan.count == 0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu);
    }

    let prefer_gpu =
        sequence_gpu_preference(plan.count, SequenceIntent::Colon, explicit_gpu).prefer_gpu;

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.linspace(start, plan.final_end, plan.count) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let data = materialize_progression(&plan, start, step);
    finalize_numeric_sequence(data, prefer_gpu)
}

fn finalize_numeric_sequence(data: Vec<f64>, prefer_gpu: bool) -> Result<Value, String> {
    let len = data.len();
    let shape = vec![1usize, len];

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|e| format!("colon: {e}"))
}

struct ProgressionPlan {
    count: usize,
    final_end: f64,
}

fn plan_progression(start: f64, step: f64, stop: f64) -> Result<ProgressionPlan, String> {
    let tol = tolerance(start, step, stop);
    let step_abs = step.abs();

    if step > 0.0 && start > stop + tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }
    if step < 0.0 && start < stop - tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let diff = (stop - start) / step;
    if !diff.is_finite() {
        return Err("colon: sequence length exceeds representable range".to_string());
    }

    let ratio_raw = (tol / step_abs).abs();
    let ratio_tol = ratio_raw
        .max(MIN_RATIO_TOL)
        .clamp(f64::EPSILON, MAX_RATIO_TOL);
    let mut approx = diff + ratio_tol;

    if approx < 0.0 {
        if approx.abs() <= ratio_tol {
            approx = 0.0;
        } else {
            return Ok(ProgressionPlan {
                count: 0,
                final_end: start,
            });
        }
    }

    if approx.is_infinite() || approx > usize::MAX as f64 {
        return Err("colon: sequence length exceeds platform limits".to_string());
    }

    let floor = approx.floor();
    let count = floor as usize;
    let count = count
        .checked_add(1)
        .ok_or_else(|| "colon: sequence length exceeds platform limits".to_string())?;

    if count == 0 {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let computed_end = start + step * ((count - 1) as f64);
    let final_end = if (computed_end - stop).abs() <= tol {
        stop
    } else {
        computed_end
    };

    Ok(ProgressionPlan { count, final_end })
}

fn materialize_progression(plan: &ProgressionPlan, start: f64, step: f64) -> Vec<f64> {
    if plan.count == 0 {
        return Vec::new();
    }
    let mut data = Vec::with_capacity(plan.count);
    for idx in 0..plan.count {
        data.push(start + step * (idx as f64));
    }
    if let Some(last) = data.last_mut() {
        *last = plan.final_end;
    }
    data
}

fn default_step(start: f64, stop: f64) -> f64 {
    if stop >= start {
        1.0
    } else {
        -1.0
    }
}

fn tolerance(start: f64, step: f64, stop: f64) -> f64 {
    let span = (stop - start).abs();
    let base = start.abs().max(stop.abs()).max(span).max(1.0);
    let step_term = step.abs().max(1.0);
    let tol = base * f64::EPSILON * 32.0 + step_term * f64::EPSILON * 16.0;
    tol.max(f64::EPSILON)
}

fn parse_real_scalar(name: &str, value: Value) -> Result<ParsedScalar, String> {
    match value {
        Value::Num(n) => ensure_finite(name, n).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Int(i) => Ok(ParsedScalar {
            value: i.to_f64(),
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Bool(b) => Ok(ParsedScalar {
            value: if b { 1.0 } else { 0.0 },
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::LogicalArray(logical) => logical_scalar(name, &logical).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Complex(re, im) => complex_to_real(name, re, im).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::CharArray(chars) => char_scalar(name, &chars).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Char,
        }),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_scalar(name, &tensor).map(|v| ParsedScalar {
                value: v,
                prefer_gpu: true,
                origin: ScalarOrigin::Numeric,
            })
        }
        Value::String(_) | Value::StringArray(_) => Err(format!(
            "{name}: inputs must be real scalar values; received a string-like argument"
        )),
        other => Err(format!(
            "{name}: inputs must be real scalar values; received {other:?}"
        )),
    }
}

fn ensure_finite(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{name}: inputs must be finite numeric scalars"))
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> Result<f64, String> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(format!("{name}: expected scalar input"));
    }
    ensure_finite(name, tensor.data[0])
}

fn logical_scalar(name: &str, logical: &LogicalArray) -> Result<f64, String> {
    if logical.len() != 1 {
        return Err(format!("{name}: expected scalar input"));
    }
    Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
}

fn complex_to_real(name: &str, re: f64, im: f64) -> Result<f64, String> {
    if im.abs() > ZERO_IM_TOL * re.abs().max(1.0) {
        return Err(format!(
            "{name}: complex inputs must have zero imaginary part"
        ));
    }
    ensure_finite(name, re)
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> Result<f64, String> {
    if tensor.data.len() != 1 {
        return Err(format!("{name}: expected scalar input"));
    }
    let (re, im) = tensor.data[0];
    complex_to_real(name, re, im)
}

fn char_scalar(name: &str, array: &CharArray) -> Result<f64, String> {
    if array.rows * array.cols != 1 {
        return Err(format!("{name}: expected scalar input"));
    }
    let ch = array.data[0];
    Ok(ch as u32 as f64)
}

fn build_char_sequence(data: Vec<f64>) -> Result<Value, String> {
    let len = data.len();
    let mut chars = Vec::with_capacity(len);
    for value in data {
        let rounded = value.round();
        if (value - rounded).abs() > CHAR_TOL {
            return Err("colon: character sequence requires integer code points".to_string());
        }
        if !(0.0..=(u32::MAX as f64)).contains(&rounded) {
            return Err("colon: character code point out of range".to_string());
        }
        let code = rounded as u32;
        let ch = std::char::from_u32(code)
            .ok_or_else(|| "colon: character code point out of range".to_string())?;
        chars.push(ch);
    }

    let array = CharArray::new(chars, 1, len).map_err(|e| format!("colon: {e}"))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, Tensor};

    #[test]
    fn colon_basic_increasing() {
        let result = colon_builtin(Value::Num(1.0), Value::Num(5.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_basic_descending() {
        let result = colon_builtin(Value::Num(5.0), Value::Num(1.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_custom_step_reaches_stop() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(0.5), vec![Value::Num(2.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_custom_step_stops_before_bound() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(2.0), vec![Value::Num(5.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![0.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_sign_mismatch_returns_empty() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(-1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_zero_increment_errors() {
        let err = colon_builtin(Value::Num(0.0), Value::Num(0.0), vec![Value::Num(1.0)])
            .expect_err("colon should error");
        assert!(err.contains("increment must be nonzero"));
    }

    #[test]
    fn colon_accepts_scalar_tensors() {
        let start = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let start_view = HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");

            let result = colon_builtin(
                Value::GpuTensor(start_handle),
                Value::Num(0.5),
                vec![Value::Num(2.0)],
            )
            .expect("colon");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 5]);
                    assert_eq!(gathered.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn colon_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let cpu = colon_builtin(Value::Num(-2.0), Value::Num(0.5), vec![Value::Num(1.0)])
            .expect("colon host");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let start = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.data,
            shape: &start.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let gpu = colon_builtin(
            Value::GpuTensor(start_handle),
            Value::Num(0.5),
            vec![Value::Num(1.0)],
        )
        .expect("colon gpu");

        let gathered = match gpu {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu")
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU result {other:?}"),
        };

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[test]
    fn colon_bool_inputs_promote() {
        let result =
            colon_builtin(Value::Bool(false), Value::Bool(true), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_char_increasing() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("e"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "abcde".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn colon_char_with_step() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let step = Value::Num(2.0);
        let stop = Value::CharArray(CharArray::new_row("g"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 4);
                let expected: Vec<char> = "aceg".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn colon_equal_endpoints_singleton() {
        let result = colon_builtin(Value::Num(3.0), Value::Num(3.0), Vec::new()).expect("colon");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0]);
            }
            other => panic!("expected scalar-compatible result, got {other:?}"),
        }
    }

    #[test]
    fn colon_complex_imaginary_errors() {
        let err = colon_builtin(Value::Complex(1.0, 1e-2), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject complex inputs");
        assert!(
            err.contains("zero imaginary part"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn colon_string_input_errors() {
        let err = colon_builtin(Value::from("hello"), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject string inputs");
        assert!(
            err.contains("string-like"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn colon_char_descending() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "fedcb".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn colon_char_fractional_step_errors() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("d"));
        let err = colon_builtin(start, Value::Num(1.5), vec![stop])
            .expect_err("colon should reject fractional char steps");
        assert!(
            err.contains("character sequence requires integer"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn colon_gpu_step_scalar_residency() {
        test_support::with_test_provider(|provider| {
            let step = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &step.data,
                shape: &step.shape,
            };
            let step_handle = provider.upload(&view).expect("upload step");
            let result = colon_builtin(
                Value::Num(0.0),
                Value::GpuTensor(step_handle),
                vec![Value::Num(2.0)],
            )
            .expect("colon");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.data, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }
}
