//! MATLAB-compatible `hypot` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    broadcast::BroadcastPlan, gpu_helpers, map_control_flow_with_builtin, tensor,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "hypot",
        builtin_path = "crate::builtins::math::elementwise::hypot"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "hypot"
category: "math/elementwise"
keywords: ["hypot", "euclidean norm", "distance", "gpu", "elementwise"]
summary: "Element-wise Euclidean norm sqrt(x.^2 + y.^2) with MATLAB-compatible broadcasting."
references: ["https://www.mathworks.com/help/matlab/ref/hypot.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs entirely on the GPU when the provider implements elem_hypot; otherwise the runtime gathers inputs and falls back to the CPU path."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::hypot::tests"
  integration: "builtins::math::elementwise::hypot::tests::hypot_gpu_provider_roundtrip"
  gpu: "builtins::math::elementwise::hypot::tests::hypot_wgpu_matches_cpu_elementwise"
---

# What does the `hypot` function do in MATLAB / RunMat?
`z = hypot(x, y)` computes the element-wise Euclidean norm `sqrt(|x|.^2 + |y|.^2)` with overflow-resistant arithmetic. It accepts scalars, vectors, matrices, N-D tensors, and complex inputs, following MATLAB's implicit expansion (broadcasting) rules.

## How does the `hypot` function behave in MATLAB / RunMat?
- Real inputs return non-negative doubles even when the operands are negative.
- Complex arguments are converted to their magnitudes before forming `sqrt(|x|.^2 + |y|.^2)`, matching MATLAB's definition since R2020b.
- Scalars broadcast across the other operand when shapes are compatible; otherwise an error is raised.
- Empty inputs propagate emptiness according to MATLAB's size rules.
- `Inf` operands return `Inf`; `NaN` operands propagate `NaN`.

## `hypot` Function GPU Execution Behaviour
When tensors already reside on the GPU, RunMat asks the active acceleration provider for the `elem_hypot` hook:

- **Hook available and shapes match:** the Euclidean norm executes in a single fused GPU kernel.
- **Hook missing or implicit expansion required:** RunMat gathers the data to the host, computes the MATLAB-compatible result, and reuses the host tensor downstream.
- **Fusion-enabled expressions:** the fusion planner can emit a WGSL `hypot(a, b)` node so long as the active provider marks the group as supported; otherwise the execution seamlessly falls back to the host path.

Providers can implement `elem_hypot` by emitting a single WGSL/compute shader that calls `hypot(a, b)`; the included WGPU backend demonstrates this pattern.

## Examples of using the `hypot` function in MATLAB / RunMat

### Computing the hypotenuse of two scalars

```matlab
result = hypot(3, 4);
```

Expected output:

```matlab
result = 5;
```

### Using `hypot` with vectors and implicit expansion

```matlab
x = [-3, 0, 4];
y = 4;
norms = hypot(x, y);
```

Expected output:

```matlab
norms = [5 4 5];
```

### Calculating per-element distances between two matrices

```matlab
X = [1 2; 3 4];
Y = [0 1; 1 0];
dist = hypot(X, Y);
```

Expected output:

```matlab
dist = [1.0000 2.2361; 3.1623 4.0000];
```

### Working with complex numbers

```matlab
a = [1+2i, 3-4i];
b = [2-1i, -1+1i];
mag = hypot(a, b);
```

Expected output:

```matlab
mag = [3.1623 5.1962];
```

### Applying `hypot` to character data

```matlab
codes = hypot('A', zeros(1, 1));
```

Expected output:

```matlab
codes = 65.0000;
```

### Executing `hypot` on GPU arrays

```matlab
Gx = gpuArray([3 4; 5 12]);
Gy = gpuArray([4 3; 12 5]);
distance_gpu = hypot(Gx, Gy);
distance = gather(distance_gpu);
```

Expected output:

```matlab
distance = [5 5; 13 13];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` calls are typically unnecessary. RunMat's planner keeps tensors on the GPU when profitable and automatically gathers data when the active provider lacks `elem_hypot`. You can still force residency with explicit `gpuArray`/`gather` to mirror MATLAB workflows or when interoperating with custom kernels.

## FAQ

### Does `hypot` overflow for large inputs?
No. Like MATLAB, RunMat uses a stable algorithm that scales the operands to avoid overflow and underflow. Very large inputs return finite results when mathematically possible.

### What happens when the inputs are negative?
`hypot` always returns a non-negative result because it squares the magnitudes first. Signs on the inputs only matter when NaN or Inf is involved.

### Can I mix real and complex inputs?
Yes. Complex inputs are converted to magnitudes before forming the norm, so you can freely combine real, complex, and logical data.

### Does `hypot` accept logical or character arrays?
Yes. Logical values map to `0` and `1`, while character arrays use their Unicode code points (as doubles), matching MATLAB semantics.

### How does broadcasting work?
The function applies MATLAB's implicit expansion: singleton dimensions expand to match the other operand. If a non-singleton dimension differs, RunMat raises a dimension mismatch error.

### Are GPU results identical to CPU results?
For double precision providers the results match bit-for-bit. Single precision providers (e.g., `wgpu` in F32 mode) may differ by typical floating-point rounding.

### What if only one operand lives on the GPU?
The runtime gathers the GPU operand, computes the norm on the host, and continues execution. Future providers may implement scalar expansion directly on the device.

### Does `hypot` allocate a new array?
Yes. The builtin returns a fresh tensor. Fusion may elide intermediate buffers when the expression participates in a larger GPU kernel.

### How can I compute the magnitude of a vector of components?
Use `hypot` repeatedly: `hypot(x, hypot(y, z))` computes the Euclidean norm of `(x, y, z)` element-wise without manual squaring.

## See Also
[sqrt](./sqrt), [abs](./abs), [sin](./sin), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for `hypot` lives at: [`crates/runmat-runtime/src/builtins/math/elementwise/hypot.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/hypot.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hypot",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_hypot",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can execute hypot in a single binary kernel; the runtime gathers to host when the hook is unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::hypot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hypot",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let a = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let b = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("hypot({a}, {b})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits WGSL hypot(a, b); providers may override via elem_hypot.",
};

const BUILTIN_NAME: &str = "hypot";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "hypot",
    category = "math/elementwise",
    summary = "Element-wise Euclidean norm sqrt(x.^2 + y.^2) with MATLAB-compatible broadcasting.",
    keywords = "hypot,euclidean norm,distance,gpu",
    accel = "binary",
    builtin_path = "crate::builtins::math::elementwise::hypot"
)]
async fn hypot_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    match (lhs, rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => hypot_gpu_pair(a, b).await,
        (Value::GpuTensor(a), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&a)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok(hypot_host(Value::Tensor(gathered), other)?)
        }
        (other, Value::GpuTensor(b)) => {
            let gathered = gpu_helpers::gather_tensor_async(&b)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok(hypot_host(other, Value::Tensor(gathered))?)
        }
        (left, right) => hypot_host(left, right),
    }
}

async fn hypot_gpu_pair(a: GpuTensorHandle, b: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if a.shape == b.shape {
            if let Ok(handle) = provider.elem_hypot(&a, &b).await {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let left = gpu_helpers::gather_tensor_async(&a)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_tensor_async(&b)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    hypot_host(Value::Tensor(left), Value::Tensor(right))
}

fn hypot_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let (Some(left), Some(right)) = (scalar_hypot_value(&lhs), scalar_hypot_value(&rhs)) {
        return Ok(Value::Num(left.hypot(right)));
    }
    let tensor_a = value_into_hypot_tensor(lhs)?;
    let tensor_b = value_into_hypot_tensor(rhs)?;
    compute_hypot_tensor(&tensor_a, &tensor_b)
}

fn compute_hypot_tensor(a: &Tensor, b: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| builtin_error(format!("hypot: {err}")))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("hypot: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut result = vec![0.0f64; plan.len()];
    for (out_idx, idx_a, idx_b) in plan.iter() {
        result[out_idx] = a.data[idx_a].hypot(b.data[idx_b]);
    }
    let tensor = Tensor::new(result, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("hypot: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn value_into_hypot_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| builtin_error(format!("hypot: {e}")))
        }
        Value::Complex(re, im) => Tensor::new(vec![complex_magnitude(re, im)], vec![1, 1])
            .map_err(|e| builtin_error(format!("hypot: {e}"))),
        Value::ComplexTensor(ct) => {
            let data: Vec<f64> = ct.data.iter().map(|(re, im)| re.hypot(*im)).collect();
            Tensor::new(data, ct.shape.clone()).map_err(|e| builtin_error(format!("hypot: {e}")))
        }
        other => {
            if let Value::GpuTensor(_) = other {
                return Err(builtin_error("hypot: internal error converting GPU tensor"));
            }
            tensor::value_into_tensor_for("hypot", other)
                .map_err(|e| builtin_error(format!("hypot: {e}")))
        }
    }
}

fn complex_magnitude(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

fn scalar_hypot_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        Value::Complex(re, im) => Some(complex_magnitude(*re, *im)),
        Value::ComplexTensor(ct) if ct.data.len() == 1 => {
            ct.data.first().map(|(re, im)| complex_magnitude(*re, *im))
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor, Value};

    fn hypot_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::hypot_builtin(lhs, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_scalar_pair() {
        let result = hypot_builtin(Value::Num(3.0), Value::Num(4.0)).expect("hypot");
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_matrix_elements() {
        let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("element-wise hypot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, 3.0, (5.0f64).sqrt(), (17.0f64).sqrt()];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_scalar_broadcast() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = hypot_builtin(Value::Tensor(matrix), Value::Num(4.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [4.123105625617661, 4.47213595499958, 5.0, 5.656854249492381];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_row_vector_broadcasts_over_matrix() {
        let matrix = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let row = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(matrix), Value::Tensor(row)).expect("row broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                let expected = [
                    (1.0f64).hypot(3.0),
                    (4.0f64).hypot(3.0),
                    (2.0f64).hypot(4.0),
                    (5.0f64).hypot(4.0),
                    (3.0f64).hypot(5.0),
                    (6.0f64).hypot(5.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_complex_scalars() {
        let left = (3.0, 4.0);
        let right = (-1.0, 2.0);
        let result = hypot_builtin(
            Value::Complex(left.0, left.1),
            Value::Complex(right.0, right.1),
        )
        .expect("complex hypot");
        let expected = complex_magnitude(left.0, left.1).hypot(complex_magnitude(right.0, right.1));
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar norm, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_complex_tensor_with_real() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (5.0, 12.0)], vec![2, 1]).unwrap();
        let real = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result =
            hypot_builtin(Value::ComplexTensor(complex), Value::Tensor(real)).expect("mixed");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected = [
                    complex_magnitude(3.0, 4.0).hypot(0.0),
                    complex_magnitude(5.0, 12.0).hypot(1.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_char_array_inputs() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = hypot_builtin(Value::CharArray(chars), Value::Int(IntValue::I32(1)))
            .expect("char hypot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [
                    (65.0f64.powi(2) + 1.0).sqrt(),
                    (66.0f64.powi(2) + 1.0).sqrt(),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_logical_inputs() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let result =
            hypot_builtin(Value::LogicalArray(logical), Value::Tensor(tensor)).expect("logical");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [
                    1.0_f64.hypot(0.0),
                    0.0_f64.hypot(1.0),
                    0.0_f64.hypot(2.0),
                    1.0_f64.hypot(3.0),
                ];
                for (actual, expect) in out.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_dimension_mismatch_errors() {
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let err = hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
        assert!(
            err.message().contains("dimension"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_nan_propagates() {
        let result = hypot_builtin(Value::Num(f64::NAN), Value::Num(1.0)).expect("nan propagation");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![3.0, 5.0, 8.0, 7.0], vec![2, 2]).unwrap();
            let rhs = Tensor::new(vec![4.0, 12.0, 15.0, 24.0], vec![2, 2]).unwrap();
            let h_lhs = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &lhs.data,
                    shape: &lhs.shape,
                })
                .expect("upload lhs");
            let h_rhs = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &rhs.data,
                    shape: &rhs.shape,
                })
                .expect("upload rhs");
            let result =
                hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
            let gathered = test_support::gather(result).expect("gathered result");
            let expected = [5.0, 13.0, 17.0, 25.0];
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_gpu_and_host_mix_falls_back() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &lhs.data,
                    shape: &lhs.shape,
                })
                .expect("upload");
            let result =
                hypot_builtin(Value::GpuTensor(handle), Value::Num(4.0)).expect("gpu + host hypot");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected: Vec<f64> = lhs.data.iter().map(|&x| x.hypot(4.0)).collect();
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hypot_empty_tensor_result() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result =
            hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("empty hypot result");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn hypot_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 12.0, 5.0], vec![2, 2]).unwrap();

        let cpu_value = compute_hypot_tensor(&lhs, &rhs).expect("cpu hypot");
        let expected = test_support::gather(cpu_value).expect("gather cpu result");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let h_lhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            })
            .expect("upload lhs");
        let h_rhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            })
            .expect("upload rhs");

        let gpu_value =
            hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");

        assert_eq!(gathered.shape, expected.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (actual, expect) in gathered.data.iter().zip(expected.data.iter()) {
            assert!(
                (actual - expect).abs() < tol,
                "|{actual} - {expect}| >= {tol}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
