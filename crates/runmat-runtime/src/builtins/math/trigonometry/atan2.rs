//! MATLAB-compatible `atan2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "atan2",
        builtin_path = "crate::builtins::math::trigonometry::atan2"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "atan2"
category: "math/trigonometry"
keywords: ["atan2", "inverse tangent", "quadrant aware", "gpu", "arctangent"]
summary: "Quadrant-aware inverse tangent atan2(y, x) with MATLAB-compatible broadcasting."
references: ["https://www.mathworks.com/help/matlab/ref/atan2.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Executes on the GPU when the provider implements elem_atan2 for shape-matched inputs; otherwise the runtime gathers operands and falls back to the host."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::atan2::tests"
  integration: "builtins::math::trigonometry::atan2::tests::atan2_gpu_provider_roundtrip"
  gpu: "builtins::math::trigonometry::atan2::tests::atan2_wgpu_matches_cpu_elementwise"
---

# What does the `atan2` function do in MATLAB / RunMat?
`theta = atan2(y, x)` computes the four-quadrant inverse tangent of the point `(x, y)`. It returns angles in radians covering the range `(-pi, pi]`, giving robust results even when `x` is zero.

## How does the `atan2` function behave in MATLAB / RunMat?
- Inputs can be scalars, vectors, matrices, or N-D tensors of real numeric, logical, or character data. MATLAB-style implicit expansion (broadcasting) applies when shapes are compatible.
- `atan2` treats `y` as the numerator and `x` as the denominator: `atan2(Y, X)` equals `atan(Y ./ X)` but keeps the correct quadrant and handles zero denominators.
- Logical inputs are cast to `double`, character arrays use their Unicode code points, and integer inputs are promoted to `double`, mirroring MATLAB promotion rules.
- Complex inputs are **not** supported; MATLAB raises an error and RunMat matches that behaviour.
- `atan2(0, 0)` returns `0`. `atan2(±0, -0)` follows IEEE-754 semantics, matching MATLAB for signed zeros and infinities.
- `atan2(NaN, x)` or `atan2(y, NaN)` returns `NaN`; inputs containing `Inf` combinations follow IEEE-754 quadrant semantics exactly like MATLAB.
- The output always has double precision and the same size as the broadcasted inputs.

## `atan2` Function GPU Execution Behaviour
- When both operands already reside on the GPU and the active provider implements the `elem_atan2` hook, RunMat executes the operation entirely on the device without reformatting buffers.
- If shapes require implicit expansion or the provider lacks `elem_atan2`, RunMat transparently gathers both tensors to the host, computes the result with the reference CPU implementation, and continues execution.
- When GPU work completes in single precision (because the provider only exposes 32-bit buffers), RunMat promotes the results to double precision whenever the data is materialised on the host so the observable behaviour still matches MATLAB.
- Fusion-aware expressions (for example, `sin(atan2(y, x))`) can still emit a combined WGSL kernel; the fusion planner recognises `atan2` as a binary elementwise primitive.

## Examples of using the `atan2` function in MATLAB / RunMat

### Computing the polar angle of a point

```matlab
theta = atan2(4, 3);
```

Expected output:

```matlab
theta = 0.9273
```

### Determining quadrants for a vector of coordinates

```matlab
Y = [-1 0 1];
X = [-1 -1 -1];
angles = atan2(Y, X);
```

Expected output:

```matlab
angles = [-2.3562 -3.1416 2.3562]
```

### Broadcasting a scalar denominator across a matrix

```matlab
A = [1 2 3; 4 5 6];
angles = atan2(A, 2);
```

Expected output:

```matlab
angles =
    0.4636    0.7854    0.9828
    1.1071    1.1903    1.2490
```

### Handling zero numerators and signed zeros

```matlab
theta = atan2([0 -0], [-2 0]);
```

Expected output:

```matlab
theta = [pi 0]
```

### Executing `atan2` on the GPU

```matlab
Gy = gpuArray([1 1; -1 -1]);
Gx = gpuArray([1 -1; 1 -1]);
angles_gpu = atan2(Gy, Gx);
angles = gather(angles_gpu);
```

Expected output:

```matlab
angles =
    0.7854    2.3562
   -0.7854   -2.3562
```

### Converting character data to angles

```matlab
theta = atan2('A', 100);
```

Expected output:

```matlab
theta = 0.5720
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat's planner keeps tensors on the GPU whenever profitable. Explicit `gpuArray` calls are optional—use them only when you need to control residency for interoperability. When `elem_atan2` is unavailable, RunMat automatically gathers data to the CPU, performs the computation, and re-uploads results only when downstream consumers demand GPU residency.

## FAQ

### What is the range of values returned by `atan2`?
Angles are given in radians and span the open/closed interval `(-pi, pi]`. Use `rad2deg` if you prefer degrees.

### Can I supply complex inputs?
No. MATLAB raises an error for complex inputs, and so does RunMat. Convert complex data to magnitude/phase first if needed.

### Does `atan2` preserve the shape of the inputs?
Yes. After implicit expansion, the output shape matches the broadcasted size of `Y` and `X`.

### How are logical or character inputs handled?
Logical values map to `0` and `1`, and character arrays use their Unicode code point values (as doubles) before computing the angle.

### What happens when `x` is zero?
`atan2` still returns a finite result using the sign of `y`. For example, `atan2(1, 0)` returns `pi/2`, and `atan2(-1, 0)` returns `-pi/2`.

### Are GPU and CPU results identical?
Double-precision providers match CPU results exactly. Single-precision providers can differ by routine IEEE rounding. RunMat automatically promotes to double when materialising host tensors to mirror MATLAB.

### How can I compute angles in degrees?
Call `rad2deg(atan2(y, x))` or multiply the result by `180/pi`.

## See Also
[atan](./atan), [hypot](./hypot), [sin](./sin), [tan](./tan), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the `atan2` builtin is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/atan2.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan2.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atan2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_atan2",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement elem_atan2 to keep the computation on device; the runtime gathers operands to the host when the hook is unavailable or broadcasting is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::atan2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atan2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let y = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let x = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("atan2({y}, {x})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits WGSL atan2(y, x); providers may override via elem_atan2 for standalone execution.",
};

#[runtime_builtin(
    name = "atan2",
    category = "math/trigonometry",
    summary = "Quadrant-aware inverse tangent atan2(y, x) with MATLAB-compatible broadcasting.",
    keywords = "atan2,inverse tangent,quadrant,gpu",
    accel = "binary",
    builtin_path = "crate::builtins::math::trigonometry::atan2"
)]
fn atan2_builtin(y: Value, x: Value) -> crate::BuiltinResult<Value> {
    match (y, x) {
        (Value::GpuTensor(yh), Value::GpuTensor(xh)) => (atan2_gpu_pair(yh, xh)).map_err(Into::into),
        (Value::GpuTensor(yh), other) => {
            let gathered = gpu_helpers::gather_tensor(&yh)?;
            atan2_host(Value::Tensor(gathered), other).map_err(Into::into)
        }
        (other, Value::GpuTensor(xh)) => {
            let gathered = gpu_helpers::gather_tensor(&xh)?;
            atan2_host(other, Value::Tensor(gathered)).map_err(Into::into)
        }
        (lhs, rhs) => (atan2_host(lhs, rhs)).map_err(Into::into),
    }
}

fn atan2_gpu_pair(y: GpuTensorHandle, x: GpuTensorHandle) -> Result<Value, String> {
    if y.device_id == x.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&y) {
            if y.shape == x.shape {
                if let Ok(handle) = provider.elem_atan2(&y, &x) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let host_y = gpu_helpers::gather_tensor(&y)?;
    let host_x = gpu_helpers::gather_tensor(&x)?;
    atan2_host(Value::Tensor(host_y), Value::Tensor(host_x))
}

fn atan2_host(y: Value, x: Value) -> Result<Value, String> {
    let tensor_y = value_into_atan2_tensor(y)?;
    let tensor_x = value_into_atan2_tensor(x)?;
    compute_atan2_tensor(&tensor_y, &tensor_x)
}

fn compute_atan2_tensor(y: &Tensor, x: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&y.shape, &x.shape)?;
    if plan.is_empty() {
        let empty = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("atan2: {e}"))?;
        return Ok(tensor::tensor_into_value(empty));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_index, idx_y, idx_x) in plan.iter() {
        out[out_index] = y.data[idx_y].atan2(x.data[idx_x]);
    }
    let tensor =
        Tensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("atan2: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn value_into_atan2_tensor(value: Value) -> Result<Tensor, String> {
    match value {
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("atan2: {e}"))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("atan2: complex inputs are not supported".to_string())
        }
        Value::GpuTensor(_) => Err("atan2: internal error converting GPU tensor".to_string()),
        other => tensor::value_into_tensor_for("atan2", other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_scalar_pair() {
        let result = atan2_builtin(Value::Num(1.0), Value::Num(1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_quadrant_detection() {
        let result = atan2_builtin(Value::Num(-1.0), Value::Num(-1.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v + 3.0 * PI / 4.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_matrix_vs_scalar_broadcast() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(2.0),
                    (2.0f64).atan2(2.0),
                    (3.0f64).atan2(2.0),
                    (4.0f64).atan2(2.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_row_vector_broadcast() {
        let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("row broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (1.0f64).atan2(1.0),
                    (-1.0f64).atan2(1.0),
                    (2.0f64).atan2(1.0),
                    (-2.0f64).atan2(1.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_char_input() {
        let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
        let result = atan2_builtin(Value::CharArray(chars), Value::Num(100.0)).expect("atan2");
        match result {
            Value::Num(v) => assert!((v - (65.0f64).atan2(100.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_logical_input() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
        let result =
            atan2_builtin(Value::LogicalArray(logical), Value::Tensor(x)).expect("logical atan2");
        match result {
            Value::Tensor(t) => {
                let expected = [
                    1.0f64.atan2(1.0),
                    0.0f64.atan2(1.0),
                    0.0f64.atan2(-1.0),
                    1.0f64.atan2(-1.0),
                ];
                for (actual, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - expect).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_zero_zero_is_zero() {
        let result = atan2_builtin(Value::Num(0.0), Value::Num(0.0)).expect("atan2");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_signed_zero_behaviour() {
        let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
        let Value::Num(pi_case) =
            atan2_builtin(Value::Num(0.0), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert!((pi_case - PI).abs() < EPS, "{pi_case} vs PI");

        let Value::Num(neg_pi_case) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(neg_zero)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert!((neg_pi_case + PI).abs() < EPS, "{neg_pi_case} vs -PI");

        let Value::Num(neg_zero_result) =
            atan2_builtin(Value::Num(neg_zero), Value::Num(0.0)).expect("atan2")
        else {
            panic!("expected numeric result");
        };
        assert_eq!(
            neg_zero_result.to_bits(),
            f64::from_bits(0x8000_0000_0000_0000).to_bits(),
            "expected negative zero, got {neg_zero_result}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_empty_tensor_result() {
        let y = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let x = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_complex_input_errors() {
        let err = atan2_builtin(Value::Complex(1.0, 1.0), Value::Num(1.0)).unwrap_err();
        assert!(err.to_ascii_lowercase().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_dimension_mismatch_errors() {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let x = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let err = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("dimension"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
            let x = Tensor::new(vec![1.0, -1.0, 1.0, -1.0], vec![2, 2]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.data,
                    shape: &y.shape,
                })
                .expect("upload y");
            let hx = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &x.data,
                    shape: &x.shape,
                })
                .expect("upload x");
            let result =
                atan2_builtin(Value::GpuTensor(hy), Value::GpuTensor(hx)).expect("gpu atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [
                (1.0f64).atan2(1.0),
                (1.0f64).atan2(-1.0),
                (-1.0f64).atan2(1.0),
                (-1.0f64).atan2(-1.0),
            ];
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn atan2_gpu_host_mix_falls_back() {
        test_support::with_test_provider(|provider| {
            let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let hy = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &y.data,
                    shape: &y.shape,
                })
                .expect("upload y");
            let result = atan2_builtin(Value::GpuTensor(hy), Value::Num(2.0)).expect("atan2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            let expected = [(1.0f64).atan2(2.0), (2.0f64).atan2(2.0)];
            for (actual, expect) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn atan2_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
        let cpu = atan2_host(Value::Tensor(y.clone()), Value::Tensor(x.clone())).unwrap();
        let hy = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            })
            .unwrap();
        let hx = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            })
            .unwrap();
        let gpu = atan2_gpu_pair(hy, hx).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expect) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!((actual - expect).abs() < tol, "{actual} vs {expect}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
