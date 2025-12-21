//! MATLAB-compatible `cos` builtin with GPU-aware semantics for RunMat.

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
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "cos"
category: "math/trigonometry"
keywords: ["cos", "cosine", "trigonometry", "elementwise", "gpu", "like"]
summary: "Cosine of scalars, vectors, matrices, complex numbers, or character arrays with MATLAB broadcasting and GPU acceleration."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_cos hooks; falls back to the host path when a provider is unavailable or cannot service the operand type."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::cos::tests"
  integration: "builtins::math::trigonometry::cos::tests::cos_gpu_provider_roundtrip"
---

# What does the `cos` function do in MATLAB / RunMat?
`y = cos(x)` evaluates the cosine of each element in `x`, interpreting angles in radians while preserving MATLAB’s column-major layout and broadcasting rules.

## How does the `cos` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors with MATLAB-compatible implicit expansion.
- Logical and integer inputs are promoted to double precision before evaluation so downstream arithmetic matches MATLAB’s numeric tower.
- Complex values use the analytic extension `cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)` while propagating `NaN`/`Inf` components independently.
- Character arrays are interpreted through their Unicode code points and return dense double arrays that mirror MATLAB’s behaviour.
- Appending `'like', prototype` mirrors the prototype’s class and residency (host or GPU), re-uploading the result when a device prototype is supplied.
- Empty inputs and singleton dimensions are preserved without introducing extraneous allocations.

## `cos` Function GPU Execution Behaviour
- With RunMat Accelerate active, tensors remain on the device and execute through the provider’s `unary_cos` hook (or fused elementwise kernels) without leaving GPU memory.
- If the provider declines the operation—for example, when only CPU precision is available or the operand type is unsupported—RunMat transparently gathers to the host, computes the result, and reapplies the requested residency rules (including `'like'` prototypes).
- Fusion planning keeps neighbouring elementwise operators grouped, reducing host↔device transfers even when an intermediate fallback occurs.

## Examples of using the `cos` function in MATLAB / RunMat

### Cosine of zero

```matlab
y = cos(0);
```

Expected output:

```matlab
y = 1
```

### Cosine of evenly spaced angles

```matlab
theta = linspace(0, 2*pi, 5);
values = cos(theta);
```

Expected output:

```matlab
values = [1 0.3090 -0.8090 -0.8090 0.3090]
```

### Cosine of complex data

```matlab
z = cos(1 + 2i);
```

Expected output:

```matlab
z = 2.0327 - 3.0519i
```

### Cosine on a GPU tensor without manual residency

```matlab
A = reshape(0:5, [3 2]);
result = cos(A);        % planner keeps the tensor on device when beneficial
```

Expected output (after `gather(result)`):

```matlab
result =
    1.0000   -0.9899
    0.5403   -0.6536
   -0.4161    0.2837
```

### Keeping results on the GPU with a `'like'` prototype

```matlab
proto = gpuArray.zeros(1, 1, 'single');
angles = [0 pi/2 pi];
deviceResult = cos(angles, 'like', proto);
```

Expected output (after `gather(deviceResult)`):

```matlab
deviceResult = single([1 0 -1])
```

### Matching a host prototype while inputs live on the GPU

```matlab
G = gpuArray([0 1 2]);
hostLike = cos(G, 'like', zeros(1, 'double'));
```

Expected output:

```matlab
hostLike = [1 0.5403 -0.4161]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` in RunMat. The fusion planner keeps tensors on the GPU whenever it is profitable and the provider exposes the required kernels. Explicit `gpuArray` / `gather` calls remain available for compatibility with MathWorks MATLAB workflows and for interacting with legacy code that relies on manual residency control.

## FAQ

### When should I use the `cos` function?
Use `cos` whenever you need the cosine of angles expressed in radians—whether those angles are scalars, vectors, matrices, or higher-dimensional tensors.

### Does `cos` promote inputs to double precision?
Yes. Unless you request otherwise with `'like'`, RunMat promotes numeric inputs to double precision, matching MATLAB’s default behaviour.

### How does `cos` handle complex inputs?
Complex numbers follow MATLAB’s analytic definition `cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)` so both the real and imaginary parts are handled correctly.

### What happens if the GPU provider lacks `unary_cos`?
RunMat gathers the tensor to the host, evaluates cosine with the CPU reference path, and then reapplies residency rules. If a `'like'` prototype targets the GPU, the result is uploaded back before returning.

### Can I rely on MATLAB broadcasting rules?
Yes. Scalar and singleton dimensions implicitly expand just as they do in MATLAB.

### Does `cos` work with character arrays?
Yes. Character arrays are converted to their Unicode code points before cosine is evaluated, and the result is returned as a dense double array of the same size.

## See Also
[sin](./sin), [tan](./tan), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `cos` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cos.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosine directly on device; runtimes gather to host when unary_cos is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cos({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cos` calls; providers can override via fused elementwise kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("cos", DOC_MD);

#[runtime_builtin(
    name = "cos",
    category = "math/trigonometry",
    summary = "Cosine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "cos,cosine,trigonometry,gpu",
    accel = "unary"
)]
fn cos_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    // Handle symbolic input
    if let Value::Symbolic(expr) = value {
        return Ok(Value::Symbolic(runmat_symbolic::SymExpr::cos(expr)));
    }

    let template = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => cos_gpu(handle)?,
        Value::Complex(re, im) => Value::Complex(cos_complex_re(re, im), cos_complex_im(re, im)),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct)?,
        Value::CharArray(ca) => cos_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err("cos: expected numeric input".to_string())
        }
        other => cos_real(other)?,
    };
    apply_output_template(base, &template)
}

fn cos_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_cos(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("cos", value)?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.cos()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("cos: {e}"))
}

fn cos_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (cos_complex_re(re, im), cos_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("cos: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cos_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cos())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("cos: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn cos_complex_re(re: f64, im: f64) -> f64 {
    re.cos() * im.cosh()
}

#[inline]
fn cos_complex_im(re: f64, im: f64) -> f64 {
    -re.sin() * im.sinh()
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
                Err("cos: expected prototype after 'like'".to_string())
            } else {
                Err("cos: unrecognised argument for cos".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("cos: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("cos: too many input arguments".to_string()),
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
                Err("cos: complex prototypes for 'like' are not supported yet".to_string())
            }
            _ => Err(
                "cos: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "cos: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("cos: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("cos: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("cos: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "cos: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("cos: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, StringArray, Tensor};

    use crate::builtins::common::test_support;

    #[test]
    fn cos_scalar_zero() {
        let result = cos_builtin(Value::Num(0.0), Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cos_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
                assert!((t.data[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cos_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cos_builtin(value, Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cos()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cos_complex_scalar() {
        let result = cos_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.cos() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im + (1.0f64.sin() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cos_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = cos_builtin(Value::CharArray(chars), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cos();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[test]
    fn cos_like_missing_prototype_errors() {
        let err =
            cos_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[test]
    fn cos_like_complex_prototype_errors() {
        let err = cos_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("complex prototypes"));
    }

    #[test]
    fn cos_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = cos_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("cos");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn cos_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("cos");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[test]
    fn cos_like_rejects_extra_arguments() {
        let err = cos_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[test]
    fn cos_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = cos_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cos()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn cos_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cos_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cos_unrecognised_option_errors() {
        let err =
            cos_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        assert!(err.contains("unrecognised argument"));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cos_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = cos_real(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = cos_gpu(h).unwrap();
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
