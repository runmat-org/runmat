//! MATLAB-compatible `acos` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse cosine with the same domain promotion, complex handling, and
//! GPU fallbacks as MATLAB. Real arguments outside `[-1, 1]` promote to complex outputs; the
//! runtime automatically gathers data to the host whenever a GPU provider cannot satisfy those
//! semantics.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const ZERO_EPS: f64 = 1e-12;
const DOMAIN_TOL: f64 = 1e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "acos",
        wasm_path = "crate::builtins::math::trigonometry::acos"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "acos"
category: "math/trigonometry"
keywords: ["acos", "inverse cosine", "arccos", "trigonometry", "gpu"]
summary: "Element-wise inverse cosine with MATLAB-compatible complex promotion and GPU fallbacks."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider's unary_acos hook when reduce_min / reduce_max confirm every element stays within [-1, 1]; gathers to host for complex promotion or when hooks are unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::acos::tests"
  integration: "builtins::math::trigonometry::acos::tests::acos_gpu_provider_roundtrip"
---

# What does the `acos` function do in MATLAB / RunMat?
`Y = acos(X)` computes the inverse cosine (in radians) of every element of `X`. Results match
MATLAB semantics for real, logical, character, and complex inputs. Real values outside the
interval `[-1, 1]` promote to complex outputs automatically so the mathematical definition remains
valid over the complex plane.

## How does the `acos` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors using MATLAB broadcasting rules.
- Logical inputs convert to double precision (`true → 1`, `false → 0`) before applying inverse cosine.
- Character arrays are interpreted as their numeric code points and return dense double or complex
  tensors that mirror MATLAB’s output.
- Real inputs with magnitude greater than `1` yield complex results (`acos(2) → 0 - 1.3170i`), matching
  MATLAB’s principal branch.
- Complex inputs follow MATLAB's definition `acos(z) = -i log(z + i sqrt(1 - z^2))`, ensuring identical
  behaviour for branch cuts and special values.
- NaNs and Infs propagate in the same way MATLAB does, including complex infinities when necessary.

## `acos` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when:

1. A provider is registered and implements `unary_acos` as well as the extremum reductions used to
   confirm the input domain.
2. Every element is provably within the real domain `[-1, 1]`.

If either condition fails (for example, when a complex result is required or the provider lacks
supporting reductions), the runtime gathers the data to the host, computes the MATLAB-compatible
answer, and returns the correct result without user intervention. Manual `gpuArray` / `gather` calls
remain optional for explicit residency control.

## Examples of using the `acos` function in MATLAB / RunMat

### Computing the inverse cosine of a scalar angle

```matlab
y = acos(0.5);
```

Expected output:

```matlab
y = 1.0472
```

### Handling values outside [-1, 1] that produce complex results

```matlab
z = acos(2);
```

Expected output:

```matlab
z = 0.0000 - 1.3170i
```

### Applying inverse cosine to every element of a matrix

```matlab
A = [0 -0.5 0.75; 1 0.25 -0.8];
Y = acos(A);
```

Expected output:

```matlab
Y =
    1.5708    2.0944    0.7227
         0    1.3181    2.4981
```

### Using logical input with automatic double promotion

```matlab
mask = logical([0 1 0 1]);
angles = acos(mask);
```

Expected output:

```matlab
angles = [1.5708 0 1.5708 0]
```

### Running `acos` on a GPU array without explicit transfers

```matlab
G = gpuArray(linspace(-1, 1, 5));
result_gpu = acos(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [3.1416 2.0944 1.5708 1.0472 0.0000]
```

### Evaluating inverse cosine of complex numbers

```matlab
vals = [0.5 + 1i, -0.2 + 0.75i];
w = acos(vals);
```

Expected output:

```matlab
w =
   1.2214 - 0.9261i
   1.7307 - 0.7009i
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the
GPU whenever the active provider exposes `unary_acos` and the values remain within the real domain.
When complex promotion is necessary, RunMat gathers the tensor automatically and returns the
MATLAB-compatible complex result. Manual residency control remains available for workflows that
require it.

## FAQ

### Why does `acos` sometimes return complex numbers?
Inputs with magnitude greater than `1` lie outside the real domain, so MATLAB (and RunMat) return
complex results computed from the principal branch of the inverse cosine.

### Does `acos` support GPU execution?
Yes. When the provider implements `unary_acos` and exposes min/max reductions for the domain check,
the runtime executes `acos` entirely on the GPU. Otherwise, it gathers to the host transparently.

### How are logical or integer inputs handled?
Logical arrays convert to doubles before evaluation. Integers also promote to double precision so
the computation matches MATLAB and avoids overflow concerns.

### What happens with NaN or Inf values?
NaNs propagate through the computation. Inputs of `±Inf` produce complex infinities exactly as in
MATLAB.

### Can I keep the result on the GPU if it becomes complex?
Complex results are currently returned on the host because GPU tensor handles represent real data.
The runtime gathers automatically and returns `Value::Complex` or `Value::ComplexTensor`, preserving
correctness.

### Does `acos` fuse with other element-wise operations?
Yes. The fusion planner can emit WGSL kernels that include `acos` when the provider supports the
generated path, allowing fused GPU execution without intermediate buffers.

## See Also
[asin](./asin), [cos](./cos), [sin](./sin), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `acos` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/acos.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/acos.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::trigonometry::acos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "acos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_acos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute acos in-place when inputs stay within [-1, 1]; otherwise the runtime gathers to host to honour MATLAB-compatible complex promotion.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::trigonometry::acos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "acos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("acos({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL acos calls; providers can substitute custom kernels when available.",
};

#[runtime_builtin(
    name = "acos",
    category = "math/trigonometry",
    summary = "Element-wise inverse cosine with MATLAB-compatible complex promotion.",
    keywords = "acos,inverse cosine,arccos,gpu",
    accel = "unary",
    wasm_path = "crate::builtins::math::trigonometry::acos"
)]
fn acos_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => acos_gpu(handle),
        Value::Complex(re, im) => Ok(acos_complex_value(re, im)),
        Value::ComplexTensor(ct) => acos_complex_tensor(ct),
        Value::CharArray(ca) => acos_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("acos: expected numeric input".to_string()),
        other => acos_real(other),
    }
}

fn acos_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_acos(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return acos_tensor_real(tensor);
            }
            Err(_) => {
                // Fall back to host path below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    acos_tensor_real(tensor)
}

fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<bool, String> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| format!("acos: reduce_min failed: {e}"))?;
    let max_handle = match provider.reduce_max(handle) {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&min_handle);
            return Err(format!("acos: reduce_max failed: {err}"));
        }
    };
    let min_host = match provider.download(&min_handle) {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(format!("acos: reduce_min download failed: {err}"));
        }
    };
    let max_host = match provider.download(&max_handle) {
        Ok(host) => host,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(format!("acos: reduce_max download failed: {err}"));
        }
    };
    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);
    if min_host.data.iter().any(|&v| v.is_nan()) || max_host.data.iter().any(|&v| v.is_nan()) {
        return Err("acos: reduction results contained NaN".to_string());
    }
    let min_val = min_host.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = max_host
        .data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(min_val < -1.0 - DOMAIN_TOL || max_val > 1.0 + DOMAIN_TOL)
}

fn acos_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("acos", value)?;
    acos_tensor_real(tensor)
}

fn acos_tensor_real(tensor: Tensor) -> Result<Value, String> {
    let len = tensor.data.len();
    if len == 0 {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut requires_complex = false;
    let mut real_data = Vec::with_capacity(len);
    let mut complex_data = Vec::with_capacity(len);

    for &v in &tensor.data {
        let result = Complex64::new(v, 0.0).acos();
        let re = zero_small(result.re);
        let im = zero_small(result.im);
        if im.abs() > ZERO_EPS {
            requires_complex = true;
        }
        real_data.push(re);
        complex_data.push((re, im));
    }

    if requires_complex {
        if len == 1 {
            let (re, im) = complex_data[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_data, tensor.shape.clone())
                .map_err(|e| format!("acos: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let tensor =
            Tensor::new(real_data, tensor.shape.clone()).map_err(|e| format!("acos: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn acos_complex_value(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).acos();
    Value::Complex(zero_small(result.re), zero_small(result.im))
}

fn acos_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    if ct.data.is_empty() {
        return Ok(Value::ComplexTensor(ct));
    }
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let result = Complex64::new(re, im).acos();
        data.push((zero_small(result.re), zero_small(result.im)));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("acos: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn acos_char_array(ca: CharArray) -> Result<Value, String> {
    if ca.data.is_empty() {
        let tensor =
            Tensor::new(Vec::new(), vec![ca.rows, ca.cols]).map_err(|e| format!("acos: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("acos: {e}"))?;
    acos_tensor_real(tensor)
}

fn zero_small(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[test]
    fn acos_scalar_within_domain() {
        let result = acos_builtin(Value::Num(0.5)).expect("acos");
        match result {
            Value::Num(v) => assert!((v - 0.5f64.acos()).abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_scalar_outside_domain_returns_complex() {
        let result = acos_builtin(Value::Num(1.2)).expect("acos");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.2, 0.0).acos();
                assert!((re - expected.re).abs() < 1e-10);
                assert!((im - expected.im).abs() < 1e-10);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_matrix_elementwise() {
        let tensor = Tensor::new(vec![0.0, -0.5, 0.75, 1.0], vec![2, 2]).expect("tensor");
        let result = acos_builtin(Value::Tensor(tensor)).expect("acos matrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0f64.acos(),
                    (-0.5f64).acos(),
                    (0.75f64).acos(),
                    1.0f64.acos(),
                ];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical");
        let result = acos_builtin(Value::LogicalArray(logical)).expect("acos logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data.len(), 4);
                assert!((t.data[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
                assert!(t.data[1].abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_char_array_complex_promotion() {
        let chars = CharArray::new("B".chars().collect(), 1, 1).expect("char");
        let result = acos_builtin(Value::CharArray(chars)).expect("acos char");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new('B' as u32 as f64, 0.0).acos();
                assert!((re - expected.re).abs() < 1e-10);
                assert!((im - expected.im).abs() < 1e-10);
            }
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.data.len(), 1);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_string_errors() {
        let err = acos_builtin(Value::from("hello")).expect_err("acos string should error");
        assert!(err.contains("expected numeric input"));
    }

    #[test]
    fn acos_integer_scalar() {
        let result = acos_builtin(Value::Int(IntValue::I32(1))).expect("acos int");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_complex_scalar_input() {
        let result = acos_builtin(Value::Complex(1.0, 2.0)).expect("acos complex");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 2.0).acos();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, -0.75, 1.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acos_builtin(Value::GpuTensor(handle)).expect("acos gpu");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            let expected = [
                0.0f64.acos(),
                0.5f64.acos(),
                (-0.75f64).acos(),
                1.0f64.acos(),
            ];
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn acos_gpu_outside_domain_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.2, -1.3], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acos_builtin(Value::GpuTensor(handle)).expect("acos gpu complex");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![2, 1]);
                }
                Value::Complex(_, _) => {}
                other => panic!("expected complex result, got {other:?}"),
            }
        });
    }

    #[test]
    fn doc_examples_present() {
        let examples = test_support::doc_examples(DOC_MD);
        assert!(!examples.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn acos_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5, 1]).unwrap();
        let cpu = acos_real(Value::Tensor(t.clone())).expect("acos cpu");
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = acos_gpu(h).expect("acos gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
