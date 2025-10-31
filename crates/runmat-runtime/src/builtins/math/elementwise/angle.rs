//! MATLAB-compatible `angle` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "angle"
category: "math/elementwise"
keywords: ["angle", "phase", "argument", "complex", "gpu"]
summary: "Phase angle of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_angle or when complex tensors require host-side conversion."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::angle::tests"
  integration: "builtins::math::elementwise::angle::tests::angle_gpu_provider_roundtrip"
---

# What does the `angle` function do in MATLAB / RunMat?
`angle(x)` returns the phase angle (argument) of each element in `x`. The values are reported in radians and match MATLAB semantics exactly.

## How does the `angle` function behave in MATLAB / RunMat?
- Complex inputs use `atan2(imag(x), real(x))`, yielding values in `[-π, π]`.
- Real inputs are treated as complex numbers with zero imaginary part. Positive numbers map to `0`, negative numbers map to `π`, and zeros return `0`.
- Logical inputs are promoted to doubles (`true → 1`, `false → 0`) before computing the angle.
- Results are always returned as dense double-precision arrays that preserve the shape of the input.
- Character arrays are interpreted as their numeric code points and produce double arrays of the same size.
- String arrays are unsupported and raise an error, mirroring MATLAB.
- NaN propagation follows IEEE rules: `angle(NaN)` returns `NaN`.

## `angle` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU remain on the device. Providers that implement the `unary_angle` hook execute the phase computation directly on the GPU, using the same `atan2(imag, real)` logic as the host path. If the hook is missing—or if the input requires host-side conversions such as complex tensors—RunMat gathers the data, evaluates the phase angle on the CPU, and then honors any downstream fusion or residency decisions automatically.

## Examples of using the `angle` function in MATLAB / RunMat

### Computing the phase of a complex scalar

```matlab
z = 3 + 4i;
theta = angle(z);
disp(theta);
```

Expected output:

```matlab
theta = 0.9273;
```

### Extracting angles from a complex vector

```matlab
Z = [1+1i, -1+1i, -1-1i, 1-1i];
phases = angle(Z);
disp(phases);
```

Expected output:

```matlab
phases = [0.7854 2.3562 -2.3562 -0.7854];
```

### Determining angles of negative real numbers

```matlab
vals = [-2 -1 0 1 2];
phi = angle(vals);
disp(phi);
```

Expected output:

```matlab
phi = [3.1416 3.1416 0 0 0];
```

### Working with GPU-resident arrays

```matlab
G = gpuArray([1+1i, -1-1i; -1+1i, 1-1i]);
thetaGPU = angle(G);
theta = gather(thetaGPU);
disp(theta);
```

Expected output:

```matlab
theta =

     0.7854   -2.3562
     2.3562    0.7854
```

When the provider implements `unary_angle`, the intermediate `thetaGPU` tensor stays resident on the device and only the explicit `gather` brings the result back.

### Angles of character code arrays

```matlab
C = ['A' 'B'; 'C' 'D'];
codes = angle(C);
disp(codes);
```

Expected output:

```matlab
codes = [0 0; 0 0];
```

### Angles from logical masks

```matlab
mask = logical([0 1; 1 0]);
phaseMask = angle(mask);
disp(phaseMask);
```

Expected output:

```matlab
phaseMask = [0 0; 0 0];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's fusion planner and Accelerate layer manage residency automatically, keeping tensors on the GPU whenever device execution is advantageous. Explicit `gpuArray` / `gather` calls remain available for MATLAB compatibility or when you need deterministic residency control.

## FAQ

### Does `angle` return values in radians?
Yes. Results are reported in radians within the interval `[-π, π]`, matching MATLAB's definition.

### What happens with zeros?
`angle(0)` returns `0`. Complex zeros (`0 + 0i`) also yield `0`.

### How does `angle` handle NaN inputs?
NaNs propagate. `angle(NaN)` evaluates to `NaN`, honoring IEEE arithmetic.

### Can I pass string arrays to `angle`?
No. Like MATLAB, `angle` supports numeric, logical, or character data. Convert strings with `double(string)` if you need numeric codes first.

### Does `angle` allocate a new array?
Yes. The builtin produces a dense double array. Fusion may eliminate the allocation when the surrounding expression can be fused safely.

### What about GPU execution for complex tensors?
Complex runtime tensors currently live on the host. RunMat gathers them automatically, computes the phase angle, and resumes execution. Future providers may add native complex kernels without changing this builtin.

### Will GPU results match CPU results exactly?
Yes for double-precision providers. Single-precision backends may exhibit minor rounding differences, but they remain within typical IEEE tolerance.

## See Also
[abs](./abs), [imag](./imag), [real](./real), [sign](./sign), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `angle` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/angle.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/angle.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "angle",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_angle" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers implement unary_angle to evaluate atan2(imag(x), real(x)) on device; the runtime gathers to host whenever the hook is unavailable or when complex tensors need host conversion.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "angle",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("atan2({zero}, {input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion assumes real-valued inputs (imaginary part zero). Complex tensors are gathered to the host until GPU complex storage lands.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("angle", DOC_MD);

#[runtime_builtin(
    name = "angle",
    category = "math/elementwise",
    summary = "Phase angle of scalars, vectors, matrices, or N-D tensors.",
    keywords = "angle,phase,argument,complex,gpu",
    accel = "unary"
)]
fn angle_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => angle_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Num(angle_scalar(re, im))),
        Value::ComplexTensor(ct) => angle_complex_tensor(ct),
        Value::CharArray(ca) => angle_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("angle: expected numeric input".to_string())
        }
        other => angle_real(other),
    }
}

fn angle_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(device_result) = provider.unary_angle(&handle) {
            return Ok(Value::GpuTensor(device_result));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    angle_tensor(tensor).map(tensor::tensor_into_value)
}

fn angle_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("angle", value)?;
    angle_tensor(tensor).map(tensor::tensor_into_value)
}

fn angle_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let Tensor { data, shape, .. } = tensor;
    let mapped: Vec<f64> = data.into_iter().map(|re| angle_scalar(re, 0.0)).collect();
    Tensor::new(mapped, shape).map_err(|e| format!("angle: {e}"))
}

fn angle_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let ComplexTensor { data, shape, .. } = ct;
    let mapped: Vec<f64> = data
        .into_iter()
        .map(|(re, im)| angle_scalar(re, im))
        .collect();
    let tensor = Tensor::new(mapped, shape).map_err(|e| format!("angle: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn angle_char_array(ca: CharArray) -> Result<Value, String> {
    let CharArray { data, rows, cols } = ca;
    let mapped: Vec<f64> = data
        .into_iter()
        .map(|ch| angle_scalar(ch as u32 as f64, 0.0))
        .collect();
    let tensor = Tensor::new(mapped, vec![rows, cols]).map_err(|e| format!("angle: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn angle_scalar(re: f64, im: f64) -> f64 {
    im.atan2(re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, StringArray};
    use std::f64::consts::PI;

    #[test]
    fn angle_real_positive_negative() {
        let pos = angle_builtin(Value::Num(5.0)).expect("angle");
        assert_eq!(pos, Value::Num(0.0));

        let neg = angle_builtin(Value::Num(-3.0)).expect("angle");
        if let Value::Num(val) = neg {
            assert!((val - PI).abs() < 1e-12);
        } else {
            panic!("expected numeric result, got {neg:?}");
        }

        let zero = angle_builtin(Value::Num(0.0)).expect("angle");
        assert_eq!(zero, Value::Num(0.0));
    }

    #[test]
    fn angle_complex_scalar_matches_atan2() {
        let value = Value::Complex(3.0, -4.0);
        let result = angle_builtin(value).expect("angle");
        if let Value::Num(angle) = result {
            assert!((angle - (-4.0f64).atan2(3.0)).abs() < 1e-12);
        } else {
            panic!("expected numeric result");
        }
    }

    #[test]
    fn angle_tensor_values() {
        let tensor = Tensor::new(vec![1.0, -1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!((out.data[0] - 0.0).abs() < 1e-12);
                assert!((out.data[1] - PI).abs() < 1e-12);
                assert_eq!(out.data[2], 0.0);
                assert_eq!(out.data[3], 0.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn angle_logical_and_char_inputs() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let logical_value = Value::LogicalArray(logical);
        let logical_result = angle_builtin(logical_value).expect("angle");
        match logical_result {
            Value::Tensor(out) => assert!(out.data.iter().all(|&v| v == 0.0)),
            other => panic!("expected tensor result, got {other:?}"),
        }

        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let char_result = angle_builtin(Value::CharArray(chars)).expect("angle");
        match char_result {
            Value::Tensor(out) => assert!(out.data.iter().all(|&v| v == 0.0)),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn angle_complex_tensor() {
        let data = vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = angle_builtin(Value::ComplexTensor(tensor)).expect("angle");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    (1.0f64).atan2(1.0),
                    (1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(-1.0),
                    (-1.0f64).atan2(1.0),
                ];
                for (actual, target) in out.data.iter().zip(expected.iter()) {
                    assert!((actual - target).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn angle_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| angle_scalar(v, 0.0)).collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, target) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - target).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn angle_dimensionless_int_input() {
        let result = angle_builtin(Value::Int(IntValue::I32(-10))).expect("angle");
        if let Value::Num(val) = result {
            assert!((val - PI).abs() < 1e-12);
        } else {
            panic!("expected numeric result");
        }
    }

    #[test]
    fn angle_nan_propagates() {
        let result = angle_builtin(Value::Num(f64::NAN)).expect("angle");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn angle_rejects_strings() {
        let err = angle_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.contains("angle: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn angle_rejects_string_arrays() {
        let array = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).unwrap();
        let err = angle_builtin(Value::StringArray(array)).unwrap_err();
        assert!(
            err.contains("angle: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn angle_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
        let cpu = angle_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = angle_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (Value::Tensor(cpu), gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
