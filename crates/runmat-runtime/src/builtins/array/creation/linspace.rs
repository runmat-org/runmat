//! MATLAB-compatible `linspace` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "linspace")]
pub const DOC_MD: &str = r#"---
title: "linspace"
category: "array/creation"
keywords: ["linspace", "range", "vector", "gpu", "sequence"]
summary: "Generate linearly spaced row vectors that match MATLAB behaviour."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider `linspace` hooks when available (implemented by the WGPU backend); otherwise falls back to uploading a host-generated sequence to keep residency intact."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::creation::linspace::tests"
  integration: "builtins::array::creation::linspace::tests::linspace_gpu_roundtrip"
  wgpu: "builtins::array::creation::linspace::tests::linspace_wgpu_matches_cpu"
---

# What does the `linspace` function do in MATLAB / RunMat?
`linspace(a, b, n)` returns `n` linearly spaced points between the start point
`a` and the end point `b`, inclusive. When `n` is omitted, MATLAB (and RunMat)
default to 100 points. The result is always a row vector.

## How does the `linspace` function behave in MATLAB / RunMat?
- `linspace(a, b)` is equivalent to `linspace(a, b, 100)`.
- `linspace(a, b, 0)` returns a `1×0` empty double row vector.
- `linspace(a, b, 1)` returns `b`.
- `linspace(a, b, 2)` returns `[a b]`.
- Complex end points are supported; the real and imaginary parts are interpolated independently.
- Inputs may be scalars, `1×1` tensors, or GPU scalars (`gpuArray` handles).
- The third argument must be a finite, non-negative integer.

## `linspace` Function GPU Execution Behaviour
If either endpoint is already resident on the GPU, RunMat keeps the result on
the device whenever the active acceleration provider supports uploading host
buffers. Providers may additionally implement the optional `linspace` hook to
generate the sequence entirely on device. Until then, RunMat generates the
sequence on the host and performs a single upload so downstream kernels can
consume the GPU data without extra gathers.

## Examples of using the `linspace` function in MATLAB / RunMat

### Creating five points between zero and one

```matlab
x = linspace(0, 1, 5)
```

Expected output:

```matlab
% Row vector with five equally spaced points
x = [0 0.25 0.5 0.75 1]
```

### Sampling 100 points when the third argument is omitted

```matlab
t = linspace(-pi, pi);
```

Expected behaviour:

```matlab
% t is a 1x100 row vector spanning [-pi, pi]
disp(t(1));
disp(t(end));
```

### Constructing complex breakpoints

```matlab
z = linspace(1+1i, -3+2i, 4);
```

Expected output:

```matlab
z =
   1.0000 + 1.0000i   -0.3333 + 1.3333i   -1.6667 + 1.6667i   -3.0000 + 2.0000i
```

### Staying on the GPU automatically

```matlab
a = gpuArray(0);
b = gpuArray(2*pi);
theta = linspace(a, b, 1024);
wave = sin(theta);
```

Expected behaviour:

```matlab
% theta and wave remain on the GPU when an acceleration provider is active.
disp(gather(theta(1:5)));
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` yourself in RunMat. When either
endpoint already resides on the GPU, RunMat keeps the generated sequence on the
device to preserve residency for downstream operations. When the active provider
implements the dedicated `linspace` hook (the WGPU backend does), the sequence is
generated directly on-device. If no acceleration provider is registered, or if a
provider lacks the hook, the runtime emits the values on the host and uploads
them once, preserving correctness with a modest transfer.

## FAQ

### What is the default number of points?
If you omit the third argument, RunMat uses 100 points, matching MATLAB.

### Does `linspace` include the end points?
Yes. The first element is exactly the start point `a` and the final element is
exactly the end point `b`.

### Can I request zero points?
Yes. `linspace(a, b, 0)` returns a `1×0` empty row vector.

### Does the third argument need to be an integer?
Yes. Non-integer or negative counts raise an error, just like MATLAB.

### Does `linspace` support complex inputs?
Yes. Real and imaginary components are interpolated independently. Mixed real
and complex inputs return a complex row vector.

### How accurate is the final point?
The implementation explicitly overwrites the final element with `b` to avoid
floating-point drift, matching MATLAB behaviour.

### Can I pass GPU scalars?
Yes. GPU endpoints are gathered once to compute the sequence. The result stays
on the GPU as long as a provider is available and the endpoints are real.

### What precision is used?
RunMat mirrors MATLAB and returns double-precision vectors. Providers may
choose to specialise future hooks for other precisions.

### How does `linspace` differ from the colon operator?
`linspace` lets you specify the number of points directly. The colon operator
(`start:step:end`) fixes the spacing instead.

### What happens when `a` equals `b`?
Every element equals `a` (and `b`). For example, `linspace(5, 5, 4)` returns
`[5 5 5 5]`.

## See Also
[zeros](./zeros), [ones](./ones), [rand](./rand), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `linspace` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/linspace.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/linspace.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linspace",
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
    notes: "Providers may generate sequences directly; the runtime uploads host-generated data when hooks are absent.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

#[runtime_builtin(
    name = "linspace",
    category = "array/creation",
    summary = "Linearly spaced vector.",
    keywords = "linspace,range,vector,gpu",
    examples = "x = linspace(0, 1, 5)  % [0 0.25 0.5 0.75 1]",
    accel = "array_construct"
)]
fn linspace_builtin(start: Value, stop: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("linspace: expected at most three input arguments".to_string());
    }

    let (start_scalar, start_gpu) = parse_scalar("linspace", start)?;
    let (stop_scalar, stop_gpu) = parse_scalar("linspace", stop)?;

    let count = if rest.is_empty() {
        100usize
    } else {
        parse_count(&rest[0])?
    };

    let residency = sequence_gpu_preference(count, SequenceIntent::Linspace, start_gpu || stop_gpu);
    if log::log_enabled!(log::Level::Trace) {
        trace!(
            "linspace: len={} prefer_gpu={} reason={:?}",
            count,
            residency.prefer_gpu,
            residency.reason
        );
    }
    let prefer_gpu = residency.prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu)
}

#[derive(Clone, Copy)]
enum Scalar {
    Real(f64),
    Complex { re: f64, im: f64 },
}

impl Scalar {
    fn parts(&self) -> (f64, f64) {
        match *self {
            Scalar::Real(r) => (r, 0.0),
            Scalar::Complex { re, im } => (re, im),
        }
    }
}

fn parse_scalar(name: &str, value: Value) -> Result<(Scalar, bool), String> {
    match value {
        Value::Num(n) => Ok((Scalar::Real(n), false)),
        Value::Int(i) => Ok((Scalar::Real(i.to_f64()), false)),
        Value::Bool(b) => Ok((Scalar::Real(if b { 1.0 } else { 0.0 }), false)),
        Value::Complex(re, im) => Ok((Scalar::Complex { re, im }, false)),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, false)),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_scalar(name, &tensor).map(|scalar| (scalar, true))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(format!(
            "{name}: endpoints must be numeric scalars; received a string-like value"
        )),
        other => Err(format!(
            "{name}: endpoints must be numeric scalars; received {other:?}"
        )),
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> Result<Scalar, String> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(format!("{name}: expected scalar input"));
    }
    Ok(Scalar::Real(tensor.data[0]))
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> Result<Scalar, String> {
    if tensor.data.len() != 1 {
        return Err(format!("{name}: expected scalar input"));
    }
    let (re, im) = tensor.data[0];
    Ok(Scalar::Complex { re, im })
}

fn parse_count(value: &Value) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err("linspace: number of points must be >= 0".to_string());
            }
            usize::try_from(raw).map_err(|_| {
                "linspace: number of points is too large for this platform".to_string()
            })
        }
        Value::Num(n) => parse_numeric_count(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err("linspace: number of points must be a scalar".to_string());
            }
            parse_numeric_count(t.data[0])
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(handle)?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err("linspace: number of points must be a scalar".to_string());
            }
            parse_numeric_count(tensor.data[0])
        }
        other => Err(format!(
            "linspace: number of points must be numeric, got {other:?}"
        )),
    }
}

fn parse_numeric_count(raw: f64) -> Result<usize, String> {
    if !raw.is_finite() {
        return Err("linspace: number of points must be finite".to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("linspace: number of points must be an integer".to_string());
    }
    if rounded < 0.0 {
        return Err("linspace: number of points must be >= 0".to_string());
    }
    if rounded > usize::MAX as f64 {
        return Err("linspace: number of points is too large for this platform".to_string());
    }
    Ok(rounded as usize)
}

fn build_sequence(
    start: Scalar,
    stop: Scalar,
    count: usize,
    prefer_gpu: bool,
) -> Result<Value, String> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start_im != 0.0 || stop_im != 0.0;

    if complex {
        let data = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
        let tensor =
            ComplexTensor::new(data, vec![1, count]).map_err(|e| format!("linspace: {e}"))?;
        return Ok(Value::ComplexTensor(tensor));
    }

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            if count > 0 {
                if log::log_enabled!(log::Level::Trace) {
                    trace!(
                        "linspace: attempting provider.linspace start={} stop={} count={}",
                        start_re,
                        stop_re,
                        count
                    );
                }
                match provider.linspace(start_re, stop_re, count) {
                    Ok(handle) => {
                        trace!("linspace: provider.linspace succeeded");
                        return Ok(Value::GpuTensor(handle));
                    }
                    Err(err) => {
                        trace!("linspace: provider.linspace failed: {err}");
                    }
                }
            }
        }
    }

    let data = generate_real_sequence(start_re, stop_re, count);
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let shape = [1usize, count];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let tensor = Tensor::new(data, vec![1, count]).map_err(|e| format!("linspace: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn generate_real_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![stop];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        data.push(start + (idx as f64) * step);
    }
    if let Some(last) = data.last_mut() {
        *last = stop;
    }
    data
}

fn generate_complex_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(stop_re, stop_im)];
    }
    let mut data = Vec::with_capacity(count);
    let step_re = (stop_re - start_re) / ((count - 1) as f64);
    let step_im = (stop_im - start_im) / ((count - 1) as f64);
    for idx in 0..count {
        let re = start_re + (idx as f64) * step_re;
        let im = start_im + (idx as f64) * step_im;
        data.push((re, im));
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[test]
    fn linspace_basic() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_default_count() {
        let result =
            linspace_builtin(Value::Num(-1.0), Value::Num(1.0), Vec::new()).expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 100]);
                assert!((t.data.first().copied().unwrap() + 1.0).abs() < 1e-12);
                assert!((t.data.last().copied().unwrap() - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_zero_count() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(10.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn linspace_single_point() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(9.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 9.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_non_integer_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(3.5)])
            .expect_err("expected error");
        assert!(err.contains("integer"));
    }

    #[test]
    fn linspace_negative_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-2))],
        )
        .expect_err("expected error");
        assert!(err.contains(">= 0"));
    }

    #[test]
    fn linspace_infinite_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.contains("finite"));
    }

    #[test]
    fn linspace_nan_count_errors() {
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect_err("expected error");
        assert!(err.contains("finite"));
    }

    #[test]
    fn linspace_non_scalar_count_errors() {
        let sz = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Tensor(sz)])
            .expect_err("expected error");
        assert!(err.contains("scalar"));
    }

    #[test]
    fn linspace_complex_sequence() {
        let result = linspace_builtin(
            Value::Complex(1.0, 1.0),
            Value::Complex(-3.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (idx, &(re, im)) in expected.iter().enumerate() {
                    let (r, i) = t.data[idx];
                    assert!((r - re).abs() < 1e-9);
                    assert!((i - im).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn linspace_boolean_arguments_are_promoted() {
        let result = linspace_builtin(
            Value::Bool(true),
            Value::Bool(false),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0, 0.5, 0.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_boolean_count_supported() {
        let result = linspace_builtin(Value::Num(3.0), Value::Num(7.0), vec![Value::Bool(true)])
            .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 7.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_tensor_scalar_arguments() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [2.0, 3.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.data[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_equal_endpoints_fill_with_endpoint() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(5.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t.data.iter().all(|v| (*v - 5.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.data,
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(5))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                    assert_eq!(gathered.shape, vec![1, 5]);
                    for (idx, expected_val) in expected.iter().enumerate() {
                        assert!((gathered.data[idx] - expected_val).abs() < 1e-12);
                    }
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn linspace_gpu_zero_count_produces_gpu_empty_vector() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.data,
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.data,
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 0]);
                    assert!(gathered.data.is_empty());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
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
    fn linspace_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        let provider =
            register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");

        let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.data,
            shape: &start.shape,
        };
        let stop_view = HostTensorView {
            data: &stop.data,
            shape: &stop.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let stop_handle = provider.upload(&stop_view).expect("upload stop");

        let result = linspace_builtin(
            Value::GpuTensor(start_handle),
            Value::GpuTensor(stop_handle),
            vec![Value::Int(IntValue::I32(9))],
        )
        .expect("linspace");
        let gathered = test_support::gather(result).expect("gather");
        let expected = generate_real_sequence(0.0, 1.0, 9);

        let precision = runmat_accelerate_api::provider()
            .expect("provider")
            .precision();
        let tol = match precision {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };

        assert_eq!(gathered.shape, vec![1, 9]);
        for (idx, expected_value) in expected.iter().enumerate() {
            let actual = gathered.data[idx];
            assert!(
                (actual - expected_value).abs() <= tol,
                "mismatch at {idx}: gpu={} expected={}",
                actual,
                expected_value
            );
        }
    }
}
