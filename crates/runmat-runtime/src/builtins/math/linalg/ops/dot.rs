//! MATLAB-compatible `dot` builtin with GPU-aware semantics for RunMat.
//!
//! Implements inner products for real and complex inputs, including dimension-aware
//! reductions that match MathWorks MATLAB behaviour. GPU inputs are gathered when
//! necessary and the result is re-uploaded to the active provider when possible so
//! downstream consumers can remain device-resident.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

const DOT_NAME: &str = "dot";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "dot",
        builtin_path = "crate::builtins::math::linalg::ops::dot"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "dot"
category: "math/linalg/ops"
keywords: ["dot", "inner product", "linear algebra", "gpu", "dimension"]
summary: "Compute dot products (inner products) of real or complex inputs, optionally along a chosen dimension."
references: ["https://www.mathworks.com/help/matlab/ref/dot.html"]
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Prefers a provider-side dot hook; the WGPU backend currently handles 2-D tensors along dimensions 1 or 2 and falls back to the host otherwise. When no hook is available RunMat gathers operands, evaluates the MATLAB reference path, and re-uploads real results when possible."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::dot::tests"
  integration: "builtins::math::linalg::ops::dot::tests::dot_gpu_roundtrip"
  doc: "builtins::math::linalg::ops::dot::tests::doc_examples_present"
---

# What does the `dot` function do in MATLAB / RunMat?
`dot(A, B)` evaluates the inner product between matching slices of `A` and `B`. Inputs must have the same size, and complex vectors follow MATLAB's convention `sum(conj(A) .* B)`.

## How does the `dot` function behave in MATLAB / RunMat?
- Treats vectors, matrices, and N-D tensors identically as long as `A` and `B` share the same size.
- When no dimension is supplied, the function reduces along the first non-singleton dimension (`dim = 1` when all dimensions are singleton).
- `dot(A, B, dim)` collapses dimension `dim` (1-based) while leaving every other dimension untouched.
- Complex inputs conjugate the first argument before multiplication; real inputs use a straight element-wise product.
- Empty reductions yield zeros of the appropriate shape; length mismatches raise `A and B must be the same size`.
- Logical and integer inputs are promoted to double precision automatically so the result is always floating point (or complex when any input is complex).

## `dot` Function GPU Execution Behaviour
RunMat Accelerate keeps GPU-resident tensors on the device whenever the active provider exposes a `dot` hook. When the hook is missing, RunMat gathers both operands to the host, evaluates the reference path, and—when the result is real—uploads it back to the provider so downstream GPU code continues without manual transfers. Complex outputs remain on the host until provider support lands.

## Examples of using the `dot` function in MATLAB / RunMat

### Computing the dot product of row vectors

```matlab
A = [1 2 3];
B = [4 5 6];
val = dot(A, B);
```

Expected output:

```matlab
val = 32
```

### Dotting column vectors to obtain a scalar

```matlab
u = [1; 3; 5];
v = [2; 4; 6];
val = dot(u, v);
```

Expected output:

```matlab
val = 44
```

### Applying `dot` along a chosen dimension

```matlab
X = [1 2 3; 4 5 6];
Y = [6 5 4; 3 2 1];
cols = dot(X, Y, 1);  % collapse rows
rows = dot(X, Y, 2);  % collapse columns
```

Expected output:

```matlab
cols = [18 20 18];
rows = [28; 28];
```

### Dotting complex vectors uses conjugation on the first input

```matlab
a = [1+2i, 3-4i];
b = [2-3i, -1+5i];
val = dot(a, b);
```

Expected output:

```matlab
val = -27 + 4i
```

### Evaluating `dot` on `gpuArray` inputs

```matlab
G1 = gpuArray([1 2 3 4]);
G2 = gpuArray([4 3 2 1]);
G = dot(G1, G2);
result = gather(G);
```

Expected output:

```matlab
isa(G, 'gpuArray')   % logical 1
result = 20
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to call `gpuArray` explicitly for `dot`. If the provider lacks a dedicated dot kernel, RunMat will gather the operands, compute the host result, and—when the output is real—upload it back to the provider automatically. Complex outputs remain on the host until GPU complex datatypes are implemented; in mixed pipelines consider reintroducing `gpuArray` explicitly after the call if residency is critical.

## FAQ

### Does `dot` require vectors?
No. Any pair of tensors with identical sizes works; specifying a dimension lets you dot slices of higher-dimensional arrays.

### How does the optional dimension behave?
`dot(A, B, dim)` collapses the `dim`th dimension (1-based). Dimensions greater than the array rank have length 1 and therefore leave the data unchanged.

### What happens with complex numbers?
The first input is conjugated before multiplication so the result matches MATLAB's hermitian inner product.

### Are empty inputs supported?
Yes. If the reduction dimension has length 0 the result is filled with zeros of the appropriate shape.

### Will the result stay on the GPU?
When a provider is active the runtime uploads real-valued results back to the device. Complex outputs stay on the host until GPU complex support is available.

### What error is raised for size mismatches?
When `A` and `B` differ in size `dot` raises: `A and B must be the same size.` matching MATLAB's wording.

### Does `dot` accept logical or integer inputs?
Yes. Inputs are promoted to double precision before evaluation so you never lose MATLAB's semantics.

### Can I request conjugation of the second argument instead?
No. MATLAB's `dot` is fixed to conjugate the first argument. Use `sum(A .* conj(B))` manually if you need the opposite orientation.

## See Also
[mtimes](./mtimes), [mldivide](./mldivide), [norm](./norm), [sum](./sum)

## Source & Feedback
- The full source code for the implementation of the `dot` function is available at: [`crates/runmat-runtime/src/builtins/math/linalg/ops/dot.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/dot.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dot",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Reduction { name: "dot" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(1024),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Dispatches to a provider-side dot implementation when available; otherwise gathers operands and re-uploads real outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::dot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Higher-level fusion currently delegates to dedicated dot kernels or host fallbacks.",
};

#[runtime_builtin(
    name = "dot",
    category = "math/linalg/ops",
    summary = "Dot product (inner product) of matching tensors along a specified dimension.",
    keywords = "dot,inner product,gpu,linear algebra",
    accel = "reduction",
    builtin_path = "crate::builtins::math::linalg::ops::dot"
)]
fn dot_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err((("dot: too many input arguments".to_string())).into());
    }
    let dim = rest
        .first()
        .map(|value| tensor::parse_dimension(value, DOT_NAME))
        .transpose()?;

    if let (Value::GpuTensor(lhs_handle), Value::GpuTensor(rhs_handle)) = (&lhs, &rhs) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            match provider.dot(lhs_handle, rhs_handle, dim) {
                Ok(handle) => return Ok(Value::GpuTensor(handle)),
                Err(err) => {
                    log::trace!("dot: provider dot fallback triggered: {err}");
                }
            }
        }
    }

    let lhs_gpu = matches!(lhs, Value::GpuTensor(_));
    let rhs_gpu = matches!(rhs, Value::GpuTensor(_));

    let lhs_host = gather_if_needed(&lhs).map_err(|e| format!("{DOT_NAME}: {e}"))?;
    let rhs_host = gather_if_needed(&rhs).map_err(|e| format!("{DOT_NAME}: {e}"))?;

    let has_complex = value_is_complex(&lhs_host) || value_is_complex(&rhs_host);

    let value = if has_complex {
        let lhs_complex = value_into_complex_tensor(lhs_host)?;
        let rhs_complex = value_into_complex_tensor(rhs_host)?;
        let result = dot_complex_tensor(&lhs_complex, &rhs_complex, dim)?;
        complex_tensor_into_value(result)
    } else {
        let lhs_tensor = tensor::value_into_tensor_for(DOT_NAME, lhs_host)?;
        let rhs_tensor = tensor::value_into_tensor_for(DOT_NAME, rhs_host)?;
        let result = dot_real_tensor(&lhs_tensor, &rhs_tensor, dim)?;
        tensor::tensor_into_value(result)
    };

    if lhs_gpu || rhs_gpu {
        promote_result_to_gpu(value)
    } else {
        Ok(value)
    }
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn value_into_complex_tensor(value: Value) -> Result<ComplexTensor, String> {
    match value {
        Value::ComplexTensor(t) => Ok(t),
        Value::Complex(re, im) => {
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("{DOT_NAME}: {e}"))
        }
        Value::Tensor(t) => real_tensor_to_complex(&t),
        Value::Num(n) => {
            let tensor =
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{DOT_NAME}: {e}"))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| format!("{DOT_NAME}: {e}"))?;
            real_tensor_to_complex(&tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("{DOT_NAME}: {e}"))?;
            real_tensor_to_complex(&tensor)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            real_tensor_to_complex(&tensor)
        }
        other => Err(format!(
            "{DOT_NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

fn real_tensor_to_complex(tensor: &Tensor) -> Result<ComplexTensor, String> {
    let shape = canonical_shape_tensor(tensor);
    let mut data = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        data.push((value, 0.0));
    }
    ComplexTensor::new(data, shape).map_err(|e| format!("{DOT_NAME}: {e}"))
}

fn dot_real_tensor(a: &Tensor, b: &Tensor, dim: Option<usize>) -> Result<Tensor, String> {
    ensure_same_size(a, b)?;

    let shape = canonical_shape_tensor(a);
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let dim_index = target_dim - 1;

    if dim_index >= shape.len() {
        return elementwise_real_product(a, b);
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let mut output = vec![0.0f64; stride_before * stride_after];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut acc = 0.0;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let prod = a.data[idx] * b.data[idx];
                acc += prod;
            }
            let out_idx = after * stride_before + before;
            output[out_idx] = acc;
        }
    }

    let mut out_shape = shape.clone();
    out_shape[dim_index] = 1;
    Tensor::new(output, out_shape).map_err(|e| format!("{DOT_NAME}: {e}"))
}

fn dot_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> Result<ComplexTensor, String> {
    ensure_same_size_complex(a, b)?;

    let shape = canonical_shape_complex(a);
    let target_dim = dim.unwrap_or_else(|| default_dimension(&shape));
    let dim_index = target_dim - 1;

    if dim_index >= shape.len() {
        return elementwise_complex_product(a, b);
    }

    let reduce_len = shape[dim_index];
    let stride_before = dim_product(&shape[..dim_index]);
    let stride_after = dim_product(&shape[dim_index + 1..]);
    let mut output = vec![(0.0f64, 0.0f64); stride_before * stride_after];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let (ar, ai) = a.data[idx];
                let (br, bi) = b.data[idx];
                let real = ar * br + ai * bi;
                let imag = ar * bi - ai * br;
                acc_re += real;
                acc_im += imag;
            }
            let out_idx = after * stride_before + before;
            output[out_idx] = (acc_re, acc_im);
        }
    }

    let mut out_shape = shape.clone();
    out_shape[dim_index] = 1;
    ComplexTensor::new(output, out_shape).map_err(|e| format!("{DOT_NAME}: {e}"))
}

pub fn dot_host_real_for_provider(
    a: &Tensor,
    b: &Tensor,
    dim: Option<usize>,
) -> Result<Tensor, String> {
    dot_real_tensor(a, b, dim)
}

pub fn dot_host_complex_for_provider(
    a: &ComplexTensor,
    b: &ComplexTensor,
    dim: Option<usize>,
) -> Result<ComplexTensor, String> {
    dot_complex_tensor(a, b, dim)
}

fn elementwise_real_product(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(a.data.len());
    for (x, y) in a.data.iter().zip(&b.data) {
        data.push(x * y);
    }
    let shape = canonical_shape_tensor(a);
    Tensor::new(data, shape).map_err(|e| format!("{DOT_NAME}: {e}"))
}

fn elementwise_complex_product(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> Result<ComplexTensor, String> {
    let mut data = Vec::with_capacity(a.data.len());
    for ((ar, ai), (br, bi)) in a.data.iter().zip(&b.data) {
        let real = ar * br + ai * bi;
        let imag = ar * bi - ai * br;
        data.push((real, imag));
    }
    let shape = canonical_shape_complex(a);
    ComplexTensor::new(data, shape).map_err(|e| format!("{DOT_NAME}: {e}"))
}

fn ensure_same_size(a: &Tensor, b: &Tensor) -> Result<(), String> {
    if a.data.len() != b.data.len() {
        return Err(format!("{DOT_NAME}: A and B must be the same size."));
    }
    if canonical_shape_tensor(a) != canonical_shape_tensor(b) {
        return Err(format!("{DOT_NAME}: A and B must be the same size."));
    }
    Ok(())
}

fn ensure_same_size_complex(a: &ComplexTensor, b: &ComplexTensor) -> Result<(), String> {
    if a.data.len() != b.data.len() {
        return Err(format!("{DOT_NAME}: A and B must be the same size."));
    }
    if canonical_shape_complex(a) != canonical_shape_complex(b) {
        return Err(format!("{DOT_NAME}: A and B must be the same size."));
    }
    Ok(())
}

fn canonical_shape_tensor(t: &Tensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn canonical_shape_complex(t: &ComplexTensor) -> Vec<usize> {
    if t.shape.is_empty() {
        vec![t.rows, t.cols]
    } else {
        t.shape.clone()
    }
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim))
}

fn promote_result_to_gpu(value: Value) -> Result<Value, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(value),
    };
    match value {
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(handle) => Ok(Value::GpuTensor(handle)),
                Err(_) => Ok(Value::Tensor(tensor)),
            }
        }
        Value::Num(n) => {
            let tensor =
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{DOT_NAME}: {e}"))?;
            promote_result_to_gpu(Value::Tensor(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            promote_result_to_gpu(Value::Tensor(tensor))
        }
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_row_vectors() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 32.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_column_vectors() {
        let lhs = Tensor::new(vec![1.0, 3.0, 5.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Num(result) => assert_eq!(result, 44.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_with_dimension_argument() {
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0], vec![2, 3]).unwrap();
        let cols = dot_builtin(
            Value::Tensor(lhs.clone()),
            Value::Tensor(rhs.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match cols {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![18.0, 20.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let rows = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("dot");
        match rows {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![28.0, 28.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_with_dimension() {
        let lhs = ComplexTensor::new(
            vec![(1.0, 1.0), (3.0, -2.0), (2.0, -3.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let rhs = ComplexTensor::new(
            vec![(2.0, -1.0), (1.0, 4.0), (-1.0, 2.0), (3.0, 5.0)],
            vec![2, 2],
        )
        .unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("dot");
        match value {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(-4.0, 11.0), (4.0, 21.0)];
                for (idx, (got, exp)) in t.data.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got.0 - exp.0).abs() < 1e-12,
                        "real mismatch at {idx}: got {}, expected {}",
                        got.0,
                        exp.0
                    );
                    assert!(
                        (got.1 - exp.1).abs() < 1e-12,
                        "imag mismatch at {idx}: got {}, expected {}",
                        got.1,
                        exp.1
                    );
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_uses_conjugate_first_argument() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -3.0), (-1.0, 5.0)], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re + 27.0).abs() < 1e-12, "expected real -27, got {re}");
                assert!((im - 4.0).abs() < 1e-12, "expected imag 4, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_complex_and_real_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value =
            dot_builtin(Value::ComplexTensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Complex(re, im) => {
                assert!((re - 11.0).abs() < 1e-12, "expected real 11, got {re}");
                assert!((im - 1.0).abs() < 1e-12, "expected imag 1, got {im}");
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_empty_reduction_returns_zero() {
        let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let value = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.0, 0.0, 0.0]);
                assert_eq!(t.shape, vec![1, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mismatched_shapes_error() {
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap();
        let err = dot_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).expect_err("dot");
        assert!(err.contains("A and B must be the same size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_zero_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect_err("expected dimension error");
        assert!(err.contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_non_integer_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Num(1.5)],
        )
        .expect_err("expected integer dimension error");
        assert!(err.contains("dimension must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_promotes_logical_inputs() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = dot_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("dot");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 7.0]);
                assert_eq!(t.shape, vec![1, 2]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let view_rhs = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::GpuTensor(gpu_rhs),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 1]);
                    assert_eq!(gathered.data, vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_mixed_gpu_and_host_returns_gpu() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
            let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![1, 4]).unwrap();
            let view_lhs = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
            let value = dot_builtin(
                Value::GpuTensor(gpu_lhs),
                Value::Tensor(rhs.clone()),
                Vec::new(),
            )
            .expect("dot");
            match value {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather result");
                    assert_eq!(gathered.shape, vec![1, 1]);
                    assert_eq!(gathered.data, vec![20.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_dimension_exceeds_rank_returns_product() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let value = dot_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::Num(3.0)],
        )
        .expect("dot");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![3.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn dot_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![6.0, 3.0, 5.0, 1.0], vec![2, 2]).unwrap();
        let cpu = dot_real_tensor(&lhs, &rhs, Some(1)).expect("cpu dot");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_lhs = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");
        let gpu_value = dot_builtin(
            Value::GpuTensor(gpu_lhs),
            Value::GpuTensor(gpu_rhs),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("gpu dot");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
