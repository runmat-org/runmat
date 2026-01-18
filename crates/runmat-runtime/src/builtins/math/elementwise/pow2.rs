//! MATLAB-compatible `pow2` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers, map_control_flow_with_builtin, tensor};

const LN_2: f64 = std::f64::consts::LN_2;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "pow2",
        builtin_path = "crate::builtins::math::elementwise::pow2"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "pow2"
category: "math/elementwise"
keywords: ["pow2", "ldexp", "binary scaling", "gpu"]
summary: "Compute 2.^X or scale mantissas by binary exponents with MATLAB-compatible semantics."
references: ["https://www.mathworks.com/help/matlab/ref/pow2.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "pow2(X) and pow2(F,E) run on the GPU when providers implement unary_pow2 and pow2_scale for matching shapes; other cases fall back to the host automatically."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::pow2::tests"
  integration: "builtins::math::elementwise::pow2::tests::pow2_gpu_roundtrip"
  gpu: "builtins::math::elementwise::pow2::tests::pow2_wgpu_matches_cpu_unary"
---

# What does the `pow2` function do in MATLAB / RunMat?
`Y = pow2(X)` computes the element-wise power-of-two `2.^X`. With two inputs, `Z = pow2(F, E)`
returns the element-wise product `F .* 2.^E`, mirroring MATLAB's `ldexp`-style scaling.

## How does the `pow2` function behave in MATLAB / RunMat?
- Accepts scalars, vectors, matrices, and N-D tensors with MATLAB's implicit expansion (broadcasting).
- Logical inputs are promoted to double before applying `2.^X`.
- Character arrays operate on their Unicode code points and return dense double tensors.
- Complex exponents yield complex outputs using the identity `2^z = exp(z * ln(2))`.
- `pow2(F, E)` supports scalar expansion on either argument and raises a dimension mismatch error when expansion is impossible.
- Empty tensors propagate emptiness with the correct MATLAB-visible shape.

## `pow2` Function GPU Execution Behaviour
When tensors already reside on the GPU, RunMat Accelerate tries the following:

1. **Unary form (`pow2(X)`):** Calls the provider hook `unary_pow2`. If the hook is unavailable,
   the runtime gathers `X`, computes on the host, and returns a CPU-resident tensor.
2. **Binary form (`pow2(F, E)`):** Calls `pow2_scale(F, E)` when both operands share identical shapes.
   Providers can implement a fused kernel (see the WGPU backend for an example). If the hook
   is missing or shapes require implicit expansion, RunMat gathers both tensors and performs the
   CPU implementation, guaranteeing MATLAB-compatible semantics.

Future providers can extend `pow2_scale` to support in-device broadcasting. Until then, fallbacks
kick in transparently without user involvement.

## Examples of using the `pow2` function in MATLAB / RunMat

### Compute power-of-two for scalar exponents

```matlab
y = pow2(3);
```

Expected output:

```matlab
y = 8;
```

### Apply `pow2` to a vector of exponents

```matlab
exponents = [-1 0 1 2];
values = pow2(exponents);
```

Expected output:

```matlab
values = [0.5 1 2 4];
```

### Scale mantissas by binary exponents

```matlab
mantissa = [0.75 1.5];
exponent = [4 5];
scaled = pow2(mantissa, exponent);
```

Expected output:

```matlab
scaled = [12 48];
```

### Use complex exponents with `pow2`

```matlab
z = pow2(1 + 2i);
```

Expected output (rounded):

```matlab
z = -0.3667 + 0.8894i;
```

### Run `pow2` on GPU arrays

```matlab
G = gpuArray([1 2 3]);
result_gpu = pow2(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [2 4 8];
```

### Convert characters to power-of-two values

```matlab
codes = pow2('ABC');
```

Expected output:

```matlab
codes = [5.9874e+41 1.2946e+42 2.7992e+42];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Explicit `gpuArray` calls are rarely needed. The acceleration planner keeps tensors on the GPU
whenever providers handle `unary_pow2` / `pow2_scale`. When hooks are missing, the runtime gathers
data, executes on the CPU, and continues seamlessly. You can still use `gpuArray` / `gather` to mirror
MATLAB workflows or to interoperate with custom kernels.

## FAQ

### Does `pow2` overflow for large exponents?
Results follow IEEE arithmetic. Very large positive exponents produce `Inf`; very negative
exponents underflow to zero.

### How are logical inputs handled?
Logical values convert to doubles (`true → 1`, `false → 0`) before applying the power.

### Can I mix scalars and arrays?
Yes. MATLAB's implicit expansion applies: singleton dimensions expand to match the other operand.

### What happens with complex inputs?
Complex exponents and/or mantissas produce complex outputs using `exp((re + i·im) * ln(2))`.

### Will GPU and CPU results differ?
Double-precision providers match CPU results bit-for-bit. Single-precision providers may differ
by expected floating-point round-off.

### Does `pow2(F,E)` allocate a new array?
Yes. The builtin returns a fresh tensor (or complex tensor). Fusion can remove intermediates
when the expression is part of a larger GPU kernel.

### Can I use `pow2` for bit shifting?
Yes. `pow2(F, E)` mirrors `ldexp`, scaling mantissas by powers of two. Integer mantissas
reproduce MATLAB's bit-shift style scaling in floating point.

## See Also
[exp](./exp), [log2](./log2), [log](./log), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/elementwise/pow2.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/pow2.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pow2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_pow2" },
        ProviderHook::Binary {
            name: "pow2_scale",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_pow2 and pow2_scale to keep tensors on-device; the runtime gathers to host when hooks are unavailable or shapes require implicit expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::pow2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pow2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input} * {:.17})", LN_2))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits `exp(x * ln2)` for unary pow2; binary scaling currently falls back to the host when implicit expansion is required.",
};

const BUILTIN_NAME: &str = "pow2";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(BUILTIN_NAME).build()
}

#[runtime_builtin(
    name = "pow2",
    category = "math/elementwise",
    summary = "Compute 2.^X or scale mantissas by binary exponents.",
    keywords = "pow2,ldexp,binary scaling,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::pow2"
)]
fn pow2_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => pow2_unary(first),
        1 => pow2_binary(first, rest.into_iter().next().unwrap()),
        _ => Err(builtin_error("pow2: expected at most two arguments")),
    }
}

fn pow2_unary(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => pow2_gpu(handle),
        Value::Complex(re, im) => {
            let (rr, ii) = pow2_complex(re, im);
            Ok(Value::Complex(rr, ii))
        }
        Value::ComplexTensor(ct) => pow2_complex_tensor(ct),
        Value::CharArray(ca) => pow2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("pow2: expected numeric input, got string"))
        }
        other => pow2_real(other),
    }
}

fn pow2_binary(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    match (mantissa, exponent) {
        (Value::GpuTensor(mh), Value::GpuTensor(eh)) => pow2_gpu_scale(mh, eh),
        (Value::GpuTensor(mh), other) => {
            let gathered = gpu_helpers::gather_tensor(&mh)
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(eh)) => {
            let gathered = gpu_helpers::gather_tensor(&eh)
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            pow2_host_scale(other, Value::Tensor(gathered))
        }
        (m, e) => pow2_host_scale(m, e),
    }
}

fn pow2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_pow2(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

fn pow2_gpu_scale(mantissa: GpuTensorHandle, exponent: GpuTensorHandle) -> BuiltinResult<Value> {
    if mantissa.device_id == exponent.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&mantissa) {
            if mantissa.shape == exponent.shape {
                if let Ok(out) = provider.pow2_scale(&mantissa, &exponent) {
                    return Ok(Value::GpuTensor(out));
                }
            }
        }
    }
    let m = gpu_helpers::gather_tensor(&mantissa)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let e = gpu_helpers::gather_tensor(&exponent)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    pow2_host_scale(Value::Tensor(m), Value::Tensor(e))
}

fn pow2_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("pow2", value)
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(tensor::tensor_into_value(pow2_tensor(tensor)?))
}

fn pow2_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = tensor.data.iter().map(|&v| v.exp2()).collect();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("pow2: {e}")))
}

fn pow2_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| pow2_complex(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn pow2_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp2())
        .collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn pow2_host_scale(mantissa: Value, exponent: Value) -> BuiltinResult<Value> {
    let mantissa_array = value_into_numeric_array(mantissa, "pow2")?;
    let exponent_array = value_into_numeric_array(exponent, "pow2")?;
    let plan = BroadcastPlan::new(mantissa_array.shape(), exponent_array.shape())
        .map_err(|e| builtin_error(format!("pow2: {e}")))?;
    if plan.is_empty() {
        if mantissa_array.is_complex() || exponent_array.is_complex() {
            let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            return Ok(Value::ComplexTensor(tensor));
        } else {
            let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            return Ok(tensor::tensor_into_value(tensor));
        }
    }
    match (mantissa_array, exponent_array) {
        (NumericArray::Real(m), NumericArray::Real(e)) => {
            let mut out = vec![0.0f64; plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let scale = e.data[idx_e].exp2();
                out[idx_out] = m.data[idx_m] * scale;
            }
            let tensor = Tensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(tensor::tensor_into_value(tensor))
        }
        (NumericArray::Real(m), NumericArray::Complex(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let (re_pow, im_pow) = pow2_complex(e.data[idx_e].0, e.data[idx_e].1);
                let scale = m.data[idx_m];
                out[idx_out] = (scale * re_pow, scale * im_pow);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        (NumericArray::Complex(m), NumericArray::Real(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let scale = e.data[idx_e].exp2();
                let (re_m, im_m) = m.data[idx_m];
                out[idx_out] = (re_m * scale, im_m * scale);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        (NumericArray::Complex(m), NumericArray::Complex(e)) => {
            let mut out = vec![(0.0f64, 0.0f64); plan.len()];
            for (idx_out, idx_m, idx_e) in plan.iter() {
                let (re_pow, im_pow) = pow2_complex(e.data[idx_e].0, e.data[idx_e].1);
                let (re_m, im_m) = m.data[idx_m];
                out[idx_out] = complex_mul(re_m, im_m, re_pow, im_pow);
            }
            let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
                .map_err(|e| builtin_error(format!("pow2: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
    }
}

fn pow2_complex(re: f64, im: f64) -> (f64, f64) {
    let scale = (re * LN_2).exp();
    let angle = im * LN_2;
    (scale * angle.cos(), scale * angle.sin())
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn value_into_numeric_array(value: Value, name: &str) -> BuiltinResult<NumericArray> {
    match value {
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(NumericArray::Complex(ct)),
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Real(tensor))
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error(format!("{name}: expected numeric input, got string")))
        }
        Value::GpuTensor(_) => Err(builtin_error(format!(
            "{name}: internal error converting GPU tensor"
        ))),
        other => {
            let tensor = tensor::value_into_tensor_for(name, other)
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Real(tensor))
        }
    }
}

enum NumericArray {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericArray {
    fn shape(&self) -> &[usize] {
        match self {
            NumericArray::Real(t) => &t.shape,
            NumericArray::Complex(t) => &t.shape,
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, NumericArray::Complex(_))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_scalar_exponent() {
        let result = pow2_builtin(Value::Num(3.0), Vec::new()).expect("pow2");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_tensor_exponent() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let result = pow2_builtin(Value::Tensor(tensor), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [0.5, 1.0, 2.0, 4.0];
                for (a, b) in out.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_scaling() {
        let mantissa = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
        let exponent = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            pow2_builtin(Value::Tensor(mantissa), vec![Value::Tensor(exponent)]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 24.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_exponent_scalar() {
        let result = pow2_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("pow2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = pow2_complex(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_complex_mantissa_real_exponent() {
        let mantissa =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -0.5)], vec![2, 1]).expect("complex tensor");
        let exponent = Tensor::new(vec![2.0, -1.0], vec![2, 1]).unwrap();
        let result = pow2_builtin(
            Value::ComplexTensor(mantissa),
            vec![Value::Tensor(exponent)],
        )
        .expect("pow2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                let scale0 = 2.0f64.exp2();
                let scale1 = (-1.0f64).exp2();
                assert!((out.data[0].0 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.data[0].1 - (1.0 * scale0)).abs() < 1e-12);
                assert!((out.data[1].0 - (2.0 * scale1)).abs() < 1e-12);
                assert!((out.data[1].1 - (-0.5 * scale1)).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_char_array() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = pow2_builtin(Value::CharArray(chars), Vec::new()).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert!((out.data[0] - (65.0f64).exp2()).abs() < 1e-6);
                assert!((out.data[1] - (66.0f64).exp2()).abs() < 1e-6);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_rejects_strings() {
        let err = pow2_builtin(Value::from("hello"), Vec::new()).unwrap_err();
        assert!(err.message().contains("expected numeric input"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pow2_builtin(Value::GpuTensor(handle), Vec::new()).expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected = vec![1.0, 2.0, 4.0];
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_gpu_scale_roundtrip() {
        test_support::with_test_provider(|provider| {
            let mantissa = Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap();
            let exponent = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let m_view = runmat_accelerate_api::HostTensorView {
                data: &mantissa.data,
                shape: &mantissa.shape,
            };
            let e_view = runmat_accelerate_api::HostTensorView {
                data: &exponent.data,
                shape: &exponent.shape,
            };
            let m_handle = provider.upload(&m_view).expect("upload m");
            let e_handle = provider.upload(&e_view).expect("upload e");
            let result = pow2_builtin(Value::GpuTensor(m_handle), vec![Value::GpuTensor(e_handle)])
                .expect("pow2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![4.0, 24.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow2_binary_broadcast_host() {
        let mantissa = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let exponent = Value::Int(IntValue::I32(2));
        let result = pow2_builtin(Value::Tensor(mantissa), vec![exponent]).expect("pow2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 8.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_matches_cpu_unary() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.5, -1.0, 0.0, 2.0, 4.25], vec![5, 1]).unwrap();
        let cpu_value = pow2_real(Value::Tensor(tensor.clone())).expect("pow2 cpu");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result from cpu path, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = pow2_gpu(handle).expect("pow2 gpu");
        let gpu = test_support::gather(gpu_value).expect("gather gpu result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (g, c) in gpu.data.iter().zip(cpu.data.iter()) {
            assert!((g - c).abs() <= tol, "mismatch: gpu={g} cpu={c} tol={tol}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pow2_wgpu_scale_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let mantissa = Tensor::new(vec![0.5, 1.5, 3.0], vec![3, 1]).unwrap();
        let exponent = Tensor::new(vec![3.0, -2.0, 5.5], vec![3, 1]).unwrap();

        let cpu_value = pow2_host_scale(
            Value::Tensor(mantissa.clone()),
            Value::Tensor(exponent.clone()),
        )
        .expect("pow2 host scale");
        let cpu = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor from cpu scale, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let m_view = runmat_accelerate_api::HostTensorView {
            data: &mantissa.data,
            shape: &mantissa.shape,
        };
        let e_view = runmat_accelerate_api::HostTensorView {
            data: &exponent.data,
            shape: &exponent.shape,
        };
        let m_handle = provider.upload(&m_view).expect("upload mantissa");
        let e_handle = provider.upload(&e_view).expect("upload exponent");
        let gpu_value = pow2_gpu_scale(m_handle, e_handle).expect("pow2 gpu scale");
        let gpu = test_support::gather(gpu_value).expect("gather gpu scale result");

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (g, c) in gpu.data.iter().zip(cpu.data.iter()) {
            assert!(
                (g - c).abs() <= tol,
                "scale mismatch: gpu={g} cpu={c} tol={tol}"
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
