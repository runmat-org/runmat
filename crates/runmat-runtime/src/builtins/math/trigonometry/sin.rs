//! MATLAB-compatible `sin` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "sin",
        builtin_path = "crate::builtins::math::trigonometry::sin"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sin"
category: "math/trigonometry"
keywords: ["sin", "sine", "trigonometry", "gpu", "elementwise", "like"]
summary: "Sine of scalars, vectors, matrices, complex numbers, or character arrays with MATLAB broadcasting and GPU acceleration."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_sin hooks; falls back to the host path when a provider is unavailable or cannot service the operand type."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::sin::tests"
  integration: "builtins::math::trigonometry::sin::tests::sin_gpu_provider_roundtrip"
---

# What does the `sin` function do in MATLAB / RunMat?
`y = sin(x)` evaluates the sine of each element in `x`, using radians and preserving MATLAB’s column-major layout and broadcasting rules.

## How does the `sin` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors with MATLAB-compatible implicit expansion.
- Logical and integer inputs are promoted to double precision before evaluation so that downstream arithmetic matches MATLAB’s numeric tower.
- Complex values use the analytic extension `sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)` while propagating `NaN`/`Inf` components independently.
- Character arrays are interpreted through their Unicode code points and return dense double arrays that mirror MATLAB’s behaviour.
- Appending `'like', prototype` mirrors the prototype’s class and residency (host or GPU), re-uploading the result when a device prototype is supplied.
- Empty inputs and singleton dimensions are preserved without introducing extraneous allocations.

## `sin` Function GPU Execution Behaviour
- With RunMat Accelerate active, tensors remain on the device and execute through the provider’s `unary_sin` hook (or fused elementwise kernels) without leaving GPU memory.
- If the provider declines the operation—for example, when only CPU precision is available or the operand type is unsupported—RunMat transparently gathers to the host, computes the result, and reapplies the requested residency rules (including `'like'` prototypes).
- Fusion planning keeps neighbouring elementwise operators grouped, reducing host↔device transfers even when an intermediate fallback occurs.

## Examples of using the `sin` function in MATLAB / RunMat

### Computing the sine of a scalar

```matlab
y = sin(pi/2);
```

Expected output:

```matlab
y = 1
```

### Computing the sine of a vector

```matlab
angles = [0 pi/6 pi/4 pi/3];
values = sin(angles);
```

Expected output:

```matlab
values = [0 0.5 0.7071 0.8660]
```

### Computing the sine of a complex number

```matlab
z = sin(1 + 2i);
```

Expected output:

```matlab
z = 3.1658 + 1.9596i
```

### Computing the sine of a matrix on a GPU

```matlab
A = reshape(0:5, [3 2]);
G = gpuArray(A);
R = sin(G);
result = gather(R);
```

Expected output:

```matlab
result =
         0    0.1411
    0.8415   -0.7568
    0.9093   -0.9589
```

### Keeping results on the GPU with a `'like'` prototype

```matlab
proto = gpuArray.zeros(1, 1, 'single');
angles = [0 pi/2];
deviceResult = sin(angles, 'like', proto);
gathered = gather(deviceResult);
```

Expected output:

```matlab
gathered =
  1x2 single
     0     1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the GPU whenever the active provider exposes the necessary kernels (such as `unary_sin`). Manual `gpuArray` / `gather` calls remain supported for MATLAB compatibility or when you need to pin residency before interacting with external code.

## FAQ

### When should I use the `sin` function?
Whenever you need the elementwise sine for signals, control systems, geometry, or any workflow that depends on periodic functions across scalars, vectors, or higher-dimensional tensors.

### Does `sin` expect radians or degrees?
Radians are required, just like in MATLAB. Use `sin(deg2rad(theta))` or the `sind` builtin if you want to work in degrees.

### How are logical, integer, or character inputs handled?
Logical and integer inputs are promoted to double precision before evaluation, and character arrays are converted to their Unicode code points. The result is always floating-point, matching MATLAB’s behaviour.

### Can the result stay on the GPU automatically?
Yes. If your inputs are already on the GPU—or you pass `'like', gpuArray(…)`—RunMat keeps the output on the device. When a fallback is required, the runtime re-uploads the result so downstream consumers still see a GPU tensor.

### What happens if the provider does not implement `unary_sin`?
RunMat gathers the tensor to the host, computes `sin` with the reference implementation, and applies the requested residency rules (including `'like'` re-uploads) before returning.

### Does `sin` support complex numbers?
Absolutely. Real and imaginary parts are handled according to the analytic extension, and NaNs/Infs propagate component-wise exactly as in MATLAB.

### Can I use complex prototypes with `'like'`?
Not yet. Provide real-valued prototypes (host or GPU) when using `'like'`; complex prototypes raise a descriptive error so you can fall back to the default output rules.

## See Also
[cos](./cos), [tan](./tan), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `sin` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sin",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sin" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sin in-place on the device; runtimes gather to host when unary_sin is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sin({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sin` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sin",
    category = "math/trigonometry",
    summary = "Sine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "sin,sine,trigonometry,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::trigonometry::sin"
)]
fn sin_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => sin_gpu(handle)?,
        Value::Complex(re, im) => Value::Complex(sin_complex_re(re, im), sin_complex_im(re, im)),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct)?,
        Value::CharArray(ca) => sin_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err("sin: expected numeric input".to_string())
        }
        other => sin_real(other)?,
    };
    apply_output_template(base, &output)
}

fn sin_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_sin(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("sin", value)?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.sin()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("sin: {e}"))
}

fn sin_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (sin_complex_re(re, im), sin_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("sin: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn sin_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sin())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("sin: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn sin_complex_re(re: f64, im: f64) -> f64 {
    re.sin() * im.cosh()
}

#[inline]
fn sin_complex_im(re: f64, im: f64) -> f64 {
    re.cos() * im.sinh()
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err("sin: expected prototype after 'like'".to_string())
            } else {
                Err("sin: unrecognised argument for sin".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("sin: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("sin: too many input arguments".to_string()),
    }
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(_) => convert_to_gpu(value),
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value),
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err("sin: complex prototypes for 'like' are not supported yet".to_string())
            }
            _ => Err(
                "sin: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "sin: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("sin: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("sin: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("sin: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "sin: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("sin: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_scalar() {
        let value = Value::Num(std::f64::consts::PI / 2.0);
        let result = sin_builtin(value, Vec::new()).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = sin_builtin(Value::Tensor(tensor), Vec::new()).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!((t.data[1] - 0.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = sin_builtin(value, Vec::new()).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.sin()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_complex_scalar() {
        let result = sin_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.sin() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im - (1.0f64.cos() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = sin_builtin(Value::CharArray(chars), Vec::new()).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sin();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sin_builtin(Value::GpuTensor(handle), Vec::new()).expect("sin");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sin()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_missing_prototype_errors() {
        let err =
            sin_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_complex_prototype_errors() {
        let err = sin_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("complex prototypes"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = sin_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("sin");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sin()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sin_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("sin");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sin()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_rejects_extra_arguments() {
        let err = sin_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = sin_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("sin");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sin()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = sin_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("sin");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
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
    fn sin_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = sin_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = sin_gpu(h).unwrap();
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
