//! MATLAB-compatible `mtimes` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{linalg, tensor};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "mtimes")]
pub const DOC_MD: &str = r#"---
title: "mtimes"
category: "math/linalg/ops"
keywords: ["mtimes", "matrix multiplication", "linear algebra", "gpu"]
summary: "Matrix multiplication (A * B) with MATLAB-compatible semantics."
references: ["https://www.mathworks.com/help/matlab/ref/mtimes.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Dispatches to the active acceleration provider via the matmul hook; otherwise gathers inputs and executes the CPU implementation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::mtimes::tests"
  integration: "builtins::math::linalg::ops::mtimes::tests::mtimes_gpu_roundtrip"
  gpu_scalar: "builtins::math::linalg::ops::mtimes::tests::gpu_scalar_matrix_product"
  wgpu: "builtins::math::linalg::ops::mtimes::tests::mtimes_wgpu_matches_cpu"
  doc: "builtins::math::linalg::ops::mtimes::tests::doc_examples_present"
---

# What does the `mtimes` function do in MATLAB / RunMat?
`mtimes(A, B)` implements MATLAB's matrix multiplication operator (`A * B`). It supports
scalars, vectors, matrices, and complex tensors while preserving MATLAB's column-major
layout and dimension rules.

## How does the `mtimes` function behave in MATLAB / RunMat?
- The inner dimensions must match: `size(A, 2) == size(B, 1)` for 2-D arrays. N-D tensors flatten the
  leading two dimensions into a matrix slice so MATLAB-style broadcasting semantics stay intact.
- Row vectors (`1×N`) times column vectors (`N×1`) evaluate to a scalar; column-by-row produces the
  familiar outer product (`N×1` · `1×M` → `N×M`).
- Scalars multiply every element of the other operand without changing shape; logical inputs are first
  converted to double precision (`true → 1`, `false → 0`).
- Complex scalars, matrices, and tensors use full complex arithmetic, including real/complex mixes that
  promote the result to a complex tensor when necessary.
- Empty matrices follow MATLAB semantics: multiplying an `m×0` by `0×n` yields an `m×n` zero matrix and
  any mismatch in inner dimensions raises `Inner matrix dimensions must agree`.
- When either input is GPU-resident, RunMat consults the active acceleration provider; if its `matmul`
  hook supports the operands the computation stays on device, otherwise the runtime gathers inputs and
  executes the CPU fallback transparently.

## `mtimes` Function GPU Execution Behaviour
1. The native auto-offload planner checks the active acceleration provider. When a backend with a
   `matmul` hook (for example, the WGPU provider) is registered, RunMat dispatches the operation there,
   keeping gpuArray inputs and the result resident on the device.
2. Mixed-residency calls automatically upload host tensors to the provider before invoking `matmul`,
   while pure scalar operands use the provider's `scalar_mul` hook to avoid unnecessary transfers.
3. If no GPU provider is registered or the backend declines the request (unsupported precision, shape,
   or size), RunMat gathers any gpuArray inputs, executes the CPU fallback in this module, and returns a
   host tensor. Reapply `gpuArray` if you need the result back on the device.

## Examples of using the `mtimes` function in MATLAB / RunMat

### Multiply two 2-D matrices

```matlab
A = [1 2 3; 4 5 6];
B = [7 8; 9 10; 11 12];
C = A * B;
```

Expected output:

```matlab
C = [58 64; 139 154];
```

### Compute a dot product with row and column vectors

```matlab
u = [1 2 3];
v = [4; 5; 6];
dotVal = u * v;
```

Expected output:

```matlab
dotVal = 32;
```

### Scale a matrix by a scalar using `mtimes`

```matlab
S = 0.5 * eye(3);
```

Expected output:

```matlab
S =
    0.5000         0         0
         0    0.5000         0
         0         0    0.5000
```

### Multiply complex matrices

```matlab
A = [1+2i 3-4i; 5+6i 7+8i];
B = [1-1i; 2+2i];
C = A * B;
```

Expected output:

```matlab
C =
   17 - 1i
    9 + 31i
```

### Perform matrix multiplication on GPU arrays

```matlab
G1 = gpuArray([1 2; 3 4]);
G2 = gpuArray([5; 6]);
G = G1 * G2;
result = gather(G);
```

Expected output:

```matlab
isa(G, 'gpuArray')   % logical 1

result =
    17
    39
```

### Dimension mismatch raises a MATLAB-style error

```matlab
A = rand(2, 3);
B = rand(4, 2);
C = A * B;
```

Expected output:

```matlab
Error using  * 
Inner matrix dimensions must agree.
```

## GPU residency in RunMat (Do I need `gpuArray`?)
When both operands already live on the GPU, the provider keeps intermediate buffers and the final
result on the device. If RunMat needs to fall back to the CPU it gathers any gpuArray inputs,
performs the multiplication, and returns a host tensor—apply `gpuArray` to the result if subsequent
steps must stay on the device. Auto-offload heuristics will continue to expand, so explicit residency
control is rarely required.

## FAQ

### How is `mtimes` different from `times` (`.*`)?
`mtimes` performs matrix multiplication (dot products, GEMM). Use `.*` for element-wise products with
implicit expansion.

### What happens when inner dimensions do not match?
RunMat raises `Inner matrix dimensions must agree`, matching MATLAB's error identifier and message.

### Does `mtimes` support scalars and matrices together?
Yes. Scalars multiply every element of the matrix, returning a matrix of the same size.

### Are complex numbers fully supported?
Yes. Mixed real/complex operands produce complex outputs using MATLAB's arithmetic rules.

### Will results stay on the GPU?
When a provider implements `matmul`, results remain device-resident. Otherwise RunMat gathers data,
computes on the CPU, and returns a host tensor.

### Do vectors need to be explicitly shaped?
Like MATLAB, row vectors must be `1×N` and column vectors `N×1`. Use `.'` or `(:)` to reshape when needed.

### Does RunMat use BLAS?
Yes. The host implementation uses RunMat's optimized inner loops today and will leverage BLAS/LAPACK
when the optional feature is enabled.

### Can `mtimes` fuse with other GPU ops?
Providers may fuse GEMM with adjacent operations; otherwise fusion falls back to the standard kernels.

## See Also
[eye](../../../array/creation/eye), [zeros](../../../array/creation/zeros), [ones](../../../array/creation/ones), [sum](../../reduction/sum), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/linalg/ops/mtimes.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/mtimes.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mtimes",
    op_kind: GpuOpKind::MatMul,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Binary {
        name: "matmul",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Calls the provider `matmul` hook when available; otherwise gathers inputs and executes the CPU fallback.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mtimes",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion currently delegates to provider matmul kernels or the CPU fallback.",
};

#[runtime_builtin(
    name = "mtimes",
    category = "math/linalg/ops",
    summary = "Matrix multiplication with MATLAB-compatible semantics.",
    keywords = "mtimes,matrix multiplication,linear algebra,gpu",
    accel = "matmul"
)]
fn mtimes_builtin(lhs: Value, rhs: Value) -> Result<Value, String> {
    mtimes_eval(&lhs, &rhs)
}

pub(crate) fn mtimes_eval(lhs: &Value, rhs: &Value) -> Result<Value, String> {
    if let Some(result) = try_gpu_matmul(lhs, rhs)? {
        return Ok(result);
    }
    mtimes_cpu(lhs.clone(), rhs.clone())
}

fn try_gpu_matmul(lhs: &Value, rhs: &Value) -> Result<Option<Value>, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }

    if !matches!(lhs, Value::GpuTensor(_)) && !matches!(rhs, Value::GpuTensor(_)) {
        return Ok(None);
    }

    if let Some(result) = try_gpu_scalar_mul(provider, lhs, rhs)? {
        return Ok(Some(result));
    }

    let mut lhs_operand = match prepare_gpu_operand(lhs, provider)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut rhs_operand = match prepare_gpu_operand(rhs, provider)? {
        Some(op) => op,
        None => {
            release_operand(provider, &mut lhs_operand);
            return Ok(None);
        }
    };

    match provider.matmul(lhs_operand.handle(), rhs_operand.handle()) {
        Ok(handle) => {
            release_operand(provider, &mut lhs_operand);
            release_operand(provider, &mut rhs_operand);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(_) => {
            release_operand(provider, &mut lhs_operand);
            release_operand(provider, &mut rhs_operand);
            Ok(None)
        }
    }
}

fn try_gpu_scalar_mul(
    provider: &'static dyn AccelProvider,
    lhs: &Value,
    rhs: &Value,
) -> Result<Option<Value>, String> {
    if let Some(scalar) = real_scalar_value(provider, lhs)? {
        if let Some(mut operand) = prepare_gpu_operand(rhs, provider)? {
            let result = provider.scalar_mul(operand.handle(), scalar);
            release_operand(provider, &mut operand);
            return match result {
                Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                Err(_) => Ok(None),
            };
        }
    }

    if let Some(scalar) = real_scalar_value(provider, rhs)? {
        if let Some(mut operand) = prepare_gpu_operand(lhs, provider)? {
            let result = provider.scalar_mul(operand.handle(), scalar);
            release_operand(provider, &mut operand);
            return match result {
                Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                Err(_) => Ok(None),
            };
        }
    }

    Ok(None)
}

fn real_scalar_value(
    provider: &'static dyn AccelProvider,
    value: &Value,
) -> Result<Option<f64>, String> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(t.data.first().copied()),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(Some(if logical.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        Value::GpuTensor(handle) if is_scalar_handle(handle) => {
            let host = provider
                .download(handle)
                .map_err(|e| format!("mtimes: {e}"))?;
            Ok(host.data.first().copied())
        }
        _ => Ok(None),
    }
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    let elements: usize = handle.shape.iter().copied().product();
    elements == 1
}

fn mtimes_cpu(lhs: Value, rhs: Value) -> Result<Value, String> {
    use Value::*;

    let lhs = crate::dispatcher::gather_if_needed(&lhs)?;
    let rhs = crate::dispatcher::gather_if_needed(&rhs)?;

    match (lhs, rhs) {
        (LogicalArray(la), other) => {
            let tensor = tensor::logical_to_tensor(&la)?;
            mtimes_cpu(Value::Tensor(tensor), other)
        }
        (other, LogicalArray(lb)) => {
            let tensor = tensor::logical_to_tensor(&lb)?;
            mtimes_cpu(other, Value::Tensor(tensor))
        }
        (Bool(b), other) => {
            let scalar = if b { 1.0 } else { 0.0 };
            mtimes_cpu(Value::Num(scalar), other)
        }
        (other, Bool(b)) => {
            let scalar = if b { 1.0 } else { 0.0 };
            mtimes_cpu(other, Value::Num(scalar))
        }
        (Complex(ar, ai), Complex(br, bi)) => Ok(Complex(ar * br - ai * bi, ar * bi + ai * br)),
        (Complex(ar, ai), Num(s)) => Ok(Complex(ar * s, ai * s)),
        (Num(s), Complex(br, bi)) => Ok(Complex(s * br, s * bi)),
        (Tensor(ta), Complex(cr, ci)) => {
            let tensor = linalg::scalar_mul_complex(&ta, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (Complex(cr, ci), Tensor(tb)) => {
            let tensor = linalg::scalar_mul_complex(&tb, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ct), Num(s)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, s, 0.0);
            Ok(complex_tensor_into_value(tensor))
        }
        (Num(s), ComplexTensor(ct)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, s, 0.0);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ct), Complex(cr, ci)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (Complex(cr, ci), ComplexTensor(ct)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ta), ComplexTensor(tb)) => {
            let tensor = linalg::matmul_complex(&ta, &tb)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ta), Tensor(tb)) => {
            let tensor = linalg::matmul_complex_real(&ta, &tb)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (Tensor(ta), ComplexTensor(tb)) => {
            let tensor = linalg::matmul_real_complex(&ta, &tb)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (Tensor(ta), Tensor(tb)) => {
            let tensor = linalg::matmul_real(&ta, &tb)?;
            Ok(tensor::tensor_into_value(tensor))
        }
        (Tensor(ta), Num(s)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(&ta, s))),
        (Num(s), Tensor(tb)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(&tb, s))),
        (Tensor(ta), Int(i)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(
            &ta,
            i.to_f64(),
        ))),
        (Int(i), Tensor(tb)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(
            &tb,
            i.to_f64(),
        ))),
        (Num(x), Num(y)) => Ok(Num(x * y)),
        (Int(x), Num(y)) => Ok(Num(x.to_f64() * y)),
        (Num(x), Int(y)) => Ok(Num(x * y.to_f64())),
        (Int(x), Int(y)) => Ok(Num(x.to_f64() * y.to_f64())),
        _ => Err("mtimes: unsupported operand types".to_string()),
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
) -> Result<Option<PreparedOperand>, String> {
    match value {
        Value::GpuTensor(handle) => {
            if is_scalar_handle(handle) {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, t)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical)?;
                let uploaded = upload_tensor(provider, &tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
) -> Result<GpuTensorHandle, String> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).map_err(|e| format!("mtimes: {e}"))?;
    Ok(handle)
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

struct PreparedOperand {
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self {
            handle,
            owned: true,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[test]
    fn matrix_product_matches_expected() {
        let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0], vec![3, 2]).unwrap();
        let result = mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![58.0, 139.0, 64.0, 154.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn scalar_matrix_product() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = mtimes_builtin(Value::Num(0.5), Value::Tensor(a)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn matrix_scalar_product() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = mtimes_builtin(Value::Tensor(a), Value::Num(3.0)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0, 6.0, 9.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn dot_product_returns_scalar() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        let result = mtimes_builtin(Value::Tensor(row), Value::Tensor(col)).expect("mtimes");
        match result {
            Value::Num(value) => assert!((value - 32.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn logical_matrix_product() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let result =
            mtimes_builtin(Value::LogicalArray(logical), Value::Tensor(matrix)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn complex_tensor_product() {
        let ct = runmat_builtins::ComplexTensor::new(
            vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, -8.0)],
            vec![2, 2],
        )
        .unwrap();
        let scalar = Value::Complex(1.0, -1.0);
        let result = mtimes_builtin(Value::ComplexTensor(ct.clone()), scalar).expect("mtimes");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, ct.shape);
                assert_eq!(
                    t.data,
                    vec![(3.0, 1.0), (-1.0, -7.0), (11.0, 1.0), (-1.0, -15.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn inner_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3, 1]).unwrap();
        let err = mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err();
        assert!(
            err.contains("Inner matrix dimensions must agree"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn mix_int_and_matrix() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            mtimes_builtin(Value::Int(IntValue::I32(2)), Value::Tensor(a)).expect("mtimes");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 6.0, 8.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn gpu_scalar_matrix_product() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mtimes_builtin(Value::Num(2.0), Value::GpuTensor(handle))
                .expect("gpu scalar matmul");
            let gathered = match result {
                Value::GpuTensor(out) => {
                    test_support::gather(Value::GpuTensor(out)).expect("gather")
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0, 6.0, 8.0]);
        });
    }

    #[test]
    fn mtimes_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");
            let result =
                mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("mtimes");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![26.0, 38.0, 30.0, 44.0]);
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn mtimes_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu =
            mtimes_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mtimes");
        let expected = test_support::gather(cpu).expect("gather cpu");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");

        let gpu = mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("wgpu mtimes");
        let gathered = test_support::gather(gpu).expect("gather gpu");

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }
}
