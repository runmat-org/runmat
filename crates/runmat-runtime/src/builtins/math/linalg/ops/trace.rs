//! MATLAB-compatible `trace` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const NAME: &str = "trace";

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = NAME)]
pub const DOC_MD: &str = r#"---
title: "trace"
category: "math/linalg/ops"
keywords: ["trace", "matrix trace", "diagonal sum", "gpu"]
summary: "Sum the diagonal elements of matrices and matrix-like tensors."
references: ["https://www.mathworks.com/help/matlab/ref/trace.html"]
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Prefers provider diag+sum hooks; otherwise gathers once, computes on the CPU, and re-uploads a 1×1 result so downstream GPU work can continue."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::trace::tests"
  integration: "builtins::math::linalg::ops::trace::tests::trace_gpu_provider_roundtrip"
  gpu: "builtins::math::linalg::ops::trace::tests::trace_wgpu_matches_cpu"
  doc: "builtins::math::linalg::ops::trace::tests::doc_examples_present"
---

# What does the `trace` function do in MATLAB / RunMat?
`trace(A)` returns the sum of the elements on the main diagonal of `A`. The result matches MATLAB
for scalars, vectors, rectangular matrices, logical masks, and complex inputs. When the argument
is a `gpuArray`, RunMat keeps the result on the GPU whenever the active provider exposes the
required hooks.

## How does the `trace` function behave in MATLAB / RunMat?
- Operates on the leading two dimensions. Higher dimensions must be singleton; otherwise an error
  is raised.
- Works for non-square matrices by summing up to `min(size(A, 1), size(A, 2))`.
- Scalars (real or complex) return their own value.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`).
- Complex inputs retain both real and imaginary parts in the result.
- Empty matrices yield `0`. Empty complex matrices yield `0 + 0i`.
- `gpuArray` inputs stay on the device when the provider implements diagonal extraction and sum
  reductions; otherwise RunMat gathers once, computes on the host, and uploads a 1×1 scalar.

## `trace` Function GPU Execution Behaviour
1. When the input already lives on the GPU and the active provider exposes both `diag_extract` and
   `reduce_sum`, RunMat extracts the diagonal on device and performs the reduction there, returning
   a `1×1` gpuArray that stays resident for downstream work.
2. If either hook is missing or the provider declines (unsupported precision, shape, or size),
   RunMat gathers the matrix exactly once, computes the diagonal sum on the CPU, and uploads the
   scalar back to the provider so subsequent GPU-friendly code keeps running on device memory.
3. Mixed-residency calls automatically upload host matrices before these steps, matching MATLAB's
   `gpuArray` behaviour while letting the auto-offload planner decide which tier benefits the most.

## Examples of using the `trace` function in MATLAB / RunMat

### Summing the diagonal of a square matrix
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
t = trace(A);
```
Expected output:
```matlab
t = 15
```

### Computing the trace of a rectangular matrix
```matlab
B = [4 2; 1 3; 5 6];
result = trace(B);
```
Expected output:
```matlab
result = 7
```

### Getting the trace of a triangular matrix
```matlab
U = [4 1 2; 0 5 3; 0 0 6];
tri_trace = trace(U);
```
Expected output:
```matlab
tri_trace = 15
```

### Working with complex-valued matrices
```matlab
Z = [1+2i 2; 3 4-5i];
zTrace = trace(Z);
```
Expected output:
```matlab
zTrace = 5.0000 - 3.0000i
```

### Tracing a gpuArray without gathering
```matlab
G = gpuArray(rand(1024));
gpuResult = trace(G);     % stays on the GPU
scalarHost = gather(gpuResult);
```
`scalarHost` is approximately `trace(rand(1024))`, and the value is computed on the GPU whenever
the provider supports diagonal extraction plus reductions.

### Handling empty matrices safely
```matlab
E = zeros(0, 5);
value = trace(E);
```
Expected output:
```matlab
value = 0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

The auto-offload planner keeps residency on the GPU when expressions benefit from it. When the
active provider exposes both `diag_extract` and `reduce_sum`, `trace` executes entirely on the GPU.
If either hook is missing, RunMat performs a single gather, computes the scalar on the CPU, and
uploads a 1×1 result back to the device so downstream fused expressions continue to operate on GPU
data.

To preserve backwards compatibility with MathWorks MATLAB—and for situations where you want to
explicitly manage residency—you can wrap inputs with `gpuArray`. This mirrors MATLAB while still
letting RunMat's planner decide whether the GPU offers an advantage for the surrounding code.

## FAQ

### What happens if my matrix is not square?
`trace` sums along the main diagonal up to `min(m, n)`, matching MATLAB behaviour for rectangular matrices.

### Does `trace` accept higher-dimensional arrays?
Only when trailing dimensions are singleton. Otherwise it raises an error because MATLAB restricts `trace` to 2-D matrix slices.

### How are logical inputs handled?
Logical values are promoted to double precision (0.0 or 1.0) before summing, mirroring MATLAB semantics.

### What is returned for empty inputs?
Empty real matrices produce `0`; empty complex matrices produce `0 + 0i`, exactly like MATLAB.

### Does the result stay on the GPU?
Yes, when the provider implements the required hooks. Otherwise RunMat re-uploads the scalar so later GPU-friendly code still sees a `gpuArray`.

### Can I call `trace` on complex data?
Absolutely. The result is a complex scalar containing the sum of the diagonal's real and imaginary parts.

### Is there any precision loss with large matrices?
`trace` accumulates in double precision (`f64`), matching MATLAB's default numeric type.

### Does `trace` modify the input matrix?
No. It reads the diagonal and returns a new scalar without altering the original matrix or its residency.

### How does `trace` interact with sparse matrices?
Sparse support is planned; current releases operate on dense arrays. Inputs are treated as dense matrices.

### Can I rely on `trace` inside fused GPU expressions?
Fused kernels treat `trace` as a scalar reduction boundary. The planner emits GPU kernels when hooks are available; otherwise it falls back gracefully.

## See Also
[diag](../../../array/shape/diag), [sum](../../reduction/sum), [mtimes](../mtimes), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/linalg/ops/trace.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/trace.rs)
- Found a behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("diag_extract"),
        ProviderHook::Reduction {
            name: "reduce_sum",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Uses provider diagonal extraction followed by a sum reduction when available; otherwise gathers once, computes on the host, and uploads a 1×1 scalar back to the device.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Trace is treated as a scalar reduction boundary; fusion wrappers stop at trace so producers/consumers can fuse independently.",
};

#[runtime_builtin(
    name = "trace",
    category = "math/linalg/ops",
    summary = "Sum the diagonal elements of matrices and matrix-like tensors.",
    keywords = "trace,matrix trace,diagonal sum,gpu",
    accel = "reduction"
)]
fn trace_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => trace_gpu(handle),
        Value::ComplexTensor(ct) => trace_complex_tensor(ct),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::CharArray(ca) => trace_char_array(ca),
        other => trace_numeric(other),
    }
}

fn trace_numeric(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for(NAME, value)?;
    ensure_matrix_shape(NAME, &tensor.shape)?;
    let sum = trace_tensor_sum(&tensor);
    Ok(Value::Num(sum))
}

fn trace_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    ensure_matrix_shape(NAME, &ct.shape)?;
    let rows = if ct.rows == 0 {
        ct.shape.first().copied().unwrap_or(0)
    } else {
        ct.rows
    };
    let cols = if ct.cols == 0 {
        if ct.shape.len() >= 2 {
            ct.shape[1]
        } else if ct.shape.len() == 1 {
            1
        } else {
            rows
        }
    } else {
        ct.cols
    };
    let diag_len = rows.min(cols);
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        let (re, im) = ct.data[linear];
        sum_re += re;
        sum_im += im;
    }
    Ok(Value::Complex(sum_re, sum_im))
}

fn trace_char_array(ca: CharArray) -> Result<Value, String> {
    ensure_matrix_shape(NAME, &[ca.rows, ca.cols])?;
    let diag_len = ca.rows.min(ca.cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx * ca.cols + idx;
        if let Some(ch) = ca.data.get(linear) {
            sum += *ch as u32 as f64;
        }
    }
    Ok(Value::Num(sum))
}

fn trace_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    ensure_matrix_shape(NAME, &handle.shape)?;
    let (rows, cols) = matrix_extents_from_shape(&handle.shape);
    let diag_len = rows.min(cols);

    if diag_len == 0 {
        return trace_gpu_fallback(&handle, 0.0);
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(diagonal) = provider.diag_extract(&handle, 0) {
            let reduced = provider.reduce_sum(&diagonal);
            let _ = provider.free(&diagonal);
            if let Ok(result) = reduced {
                return Ok(Value::GpuTensor(result));
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let sum = trace_tensor_sum(&tensor);
    trace_gpu_fallback(&handle, sum)
}

fn trace_gpu_fallback(_handle: &GpuTensorHandle, sum: f64) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let data = vec![sum];
        let shape = [1usize, 1usize];
        if let Ok(h) = provider.upload(&HostTensorView {
            data: &data,
            shape: &shape,
        }) {
            return Ok(Value::GpuTensor(h));
        }
    }
    // If no provider is registered, return a host scalar
    Ok(Value::Num(sum))
}

fn trace_tensor_sum(tensor: &Tensor) -> f64 {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let diag_len = rows.min(cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        sum += tensor.data[linear];
    }
    sum
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> Result<(), String> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(format!("{name}: input must be 2-D"))
    } else {
        Ok(())
    }
}

fn matrix_extents_from_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Tensor};

    #[test]
    fn trace_scalar_num() {
        let result = trace_builtin(Value::Num(7.0)).expect("trace");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn trace_rectangular_matrix() {
        let tensor = Tensor::new(vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0], vec![3, 2]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(10.0));
    }

    #[test]
    fn trace_vector_returns_first_element() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn trace_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 5]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn trace_complex_matrix() {
        let data = vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, 8.0)];
        let ct = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = trace_builtin(Value::ComplexTensor(ct)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 8.0).abs() < 1e-12);
                assert!((im - 10.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn trace_char_array_promotes_to_double() {
        let chars = CharArray::new("ab".chars().collect(), 1, 2).unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => assert!((value - ('a' as u32 as f64)).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn trace_char_array_square_matrix_uses_diagonal() {
        let chars = CharArray::new("abcd".chars().collect(), 2, 2).unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => {
                let expected = ('a' as u32 as f64) + ('d' as u32 as f64);
                assert!((value - expected).abs() < 1e-12);
            }
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn trace_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = provider.download(&out).expect("download");
                    assert_eq!(host.shape, vec![1, 1]);
                    assert_eq!(host.data.len(), 1);
                    assert!((host.data[0] - 6.0).abs() < 1e-12);
                    let _ = provider.free(&out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[test]
    fn trace_gpu_fallback_uploads_scalar() {
        // Force gather path by using a zero-length diagonal
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = provider.download(&out).expect("download");
                    assert_eq!(host.data, vec![0.0]);
                    let _ = provider.free(&out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[test]
    fn trace_integer_promotes_to_double() {
        let value = Value::Int(IntValue::I32(5));
        let result = trace_builtin(value).expect("trace");
        assert_eq!(result, Value::Num(5.0));
    }

    #[test]
    fn trace_bool_promotes_to_double() {
        let result = trace_builtin(Value::Bool(true)).expect("trace");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn trace_logical_array_matches_numeric() {
        let data = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let logical = LogicalArray::new(data, vec![3, 3]).expect("logical");
        let result = trace_builtin(Value::LogicalArray(logical)).expect("trace");
        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn trace_complex_empty_matrix_returns_zero() {
        let complex = ComplexTensor::new(Vec::new(), vec![0, 5]).expect("complex");
        let result = trace_builtin(Value::ComplexTensor(complex)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 0.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex zero, got {other:?}"),
        }
    }

    #[test]
    fn trace_rejects_higher_dimensional_inputs() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = trace_builtin(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err, "trace: input must be 2-D");
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn trace_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 8.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let cpu = trace_numeric(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = trace_builtin(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let expected = match cpu {
            Value::Num(n) => n,
            Value::Tensor(t) if !t.data.is_empty() => t.data[0],
            Value::Tensor(_) => 0.0,
            other => panic!("unexpected cpu comparison value {other:?}"),
        };
        assert_eq!(gathered.shape, vec![1, 1]);
        let actual = gathered
            .data
            .first()
            .copied()
            .expect("gathered tensor should contain one element");
        assert!((expected - actual).abs() < 1e-9);
    }
}
