//! MATLAB-compatible `tan` builtin with GPU-aware semantics for RunMat.

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
        name = "tan",
        wasm_path = "crate::builtins::math::trigonometry::tan"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tan"
category: "math/trigonometry"
keywords: ["tan", "tangent", "trigonometry", "radians", "gpu", "like"]
summary: "Element-wise tangent for scalars, vectors, matrices, complex numbers, or character arrays with MATLAB broadcasting and GPU acceleration."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_tan hooks; falls back to the host path when the hook is unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::tan::tests"
  integration: "builtins::math::trigonometry::tan::tests::tan_gpu_provider_roundtrip"
---

# What does the `tan` function do in MATLAB / RunMat?
`y = tan(x)` computes the tangent of every element in `x`, interpreting each input value in radians.

## How does the `tan` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors while respecting MATLAB's implicit-expansion (broadcasting) rules.
- Logical and integer inputs are promoted to double precision (`true → 1.0`, `false → 0.0`) before the tangent is applied so downstream arithmetic mirrors MATLAB.
- Character arrays operate on their Unicode code points and return dense double arrays of identical shape.
- Complex inputs follow MATLAB’s analytic extension `tan(a + bi) = sin(2a)/(cos(2a) + cosh(2b)) + i·sinh(2b)/(cos(2a) + cosh(2b))`, propagating `NaN`/`Inf` components independently.
- Appending `'like', prototype` mirrors the prototype’s class and residency for real numeric prototypes; complex prototypes currently raise a descriptive error so you can fall back to the default output rules.
- `tan` accepts both host tensors and `gpuArray` inputs. GPU residency is preserved whenever the active provider exposes the `unary_tan` kernel or a fused elementwise implementation.
- Strings and string arrays are rejected to match MATLAB’s numeric-only contract for the trigonometric family.
- Empty inputs, singleton dimensions, and already-reduced shapes pass through unchanged to avoid unnecessary allocations.

## `tan` Function GPU Execution Behaviour
- When RunMat Accelerate is active and the selected provider implements `unary_tan`, the operation executes entirely on the GPU, keeping inputs and outputs resident in device memory.
- If the provider declines the request, RunMat gathers the data to the host, evaluates the tangent with the reference implementation, and reapplies any residency requests (including `'like'` prototypes that expect GPU outputs) before returning.
- Fusion planning groups neighbouring elementwise operators, so even when a fallback occurs, subsequent GPU-capable steps can resume on-device without redundant transfers.

## Examples of using the `tan` function in MATLAB / RunMat

### Getting the tangent of 45 degrees expressed in radians

```matlab
y = tan(pi/4);
```

Expected output:

```matlab
y = 1
```

### Applying tangent to a vector of sample angles

```matlab
theta = linspace(-pi/2 + 0.1, pi/2 - 0.1, 5);
wave = tan(theta);
```

Expected output (approximate):

```matlab
wave = [-9.9666   -0.9047         0    0.9047    9.9666]
```

### Evaluating the tangent of a matrix on the GPU

```matlab
G = gpuArray([0 pi/6; pi/4 pi/3]);
T = tan(G);
result = gather(T);
```

Expected output:

```matlab
result =
         0    0.5774
    1.0000    1.7321
```

### Computing the tangent of complex angles

```matlab
z = 1 + 0.5i;
tz = tan(z);
```

Expected output (approximate):

```matlab
tz = 0.9654 + 0.2718i
```

### Using tangent to linearise small-angle approximations

```matlab
eps = [-1e-6 0 1e-6];
approx = tan(eps);
```

Expected output:

```matlab
approx = [-1.0000e-06         0    1.0000e-06]
```

### Converting degrees to radians before using `tan`

```matlab
angles_in_deg = [0 30 60 89];
radians = deg2rad(angles_in_deg);
result = tan(radians);
```

Expected output (approximate):

```matlab
result = [0    0.5774    1.7321   57.2899]
```

### Keeping the result on the GPU with `'like'`

```matlab
proto = gpuArray.zeros(1, 1, 'single');
angles = gpuArray([0 pi/6 pi/4]);
deviceResult = tan(angles, 'like', proto);
gathered = gather(deviceResult);
```

Expected output:

```matlab
gathered =
  1x3 single
         0    0.5774    1.0000
```

### Inspecting character codes with tangent

```matlab
codes = tan('ABC');
```

Expected output (approximate):

```matlab
codes =
   -1.4700    0.0266    1.6523
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the GPU whenever the active provider exposes the necessary kernels (such as `unary_tan`). Manual `gpuArray` / `gather` calls remain supported for MATLAB compatibility or when you need to pin residency before interacting with external code.

## FAQ

### When should I use the `tan` function?
Use `tan` whenever you need the tangent of angles expressed in radians—whether you are modelling waves, evaluating transfer functions, or analysing geometric relationships element-wise across large arrays.

### What happens near odd multiples of `pi/2`?
Values that approach `(2k + 1)·pi/2` grow rapidly toward ±Inf, just like in MATLAB. RunMat preserves IEEE semantics, so you may observe very large magnitudes or `Inf`/-`Inf` where poles occur.

### Does `tan` support complex numbers?
Yes. RunMat evaluates the analytic extension described above, handling real and imaginary components independently and propagating `NaN`/`Inf` values exactly as MATLAB does.

### Can I keep the result on the GPU?
Yes. When a provider implements `unary_tan`, the result never leaves the device. If a fallback is required, RunMat gathers to the host, computes the result, and reapplies residency requests—including `'like'` outputs—before handing the value back.

### How do I control units?
`tan` operates in radians. Convert degrees with `deg2rad` (or multiply by `pi/180`) before calling `tan`, or use MATLAB’s degree-specific builtins (`tand`, etc.) when available.

### Does `tan` change the input type?
By default results are double precision. Passing `'like', prototype` preserves the prototype’s real numeric type (host or GPU). Complex prototypes are currently unsupported and raise a clear error.

### Are logical arrays supported?
Yes. Logical arrays are promoted to `0.0`/`1.0` doubles before evaluation so downstream workflows remain compatible with MATLAB semantics.

### Can I fuse `tan` with other elementwise operations?
Definitely. The fusion planner emits WGSL `tan` calls for GPU execution, and providers may specialise fused kernels for additional throughput.

## See Also
[sin](./sin), [cos](./cos), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `tan` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tan.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::trigonometry::tan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tan" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute tan in place via unary_tan; runtimes gather to host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::trigonometry::tan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tan({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL tan calls; providers can override with optimised fused kernels.",
};

#[runtime_builtin(
    name = "tan",
    category = "math/trigonometry",
    summary = "Tangent of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "tan,tangent,trigonometry,radians,gpu",
    accel = "unary",
    wasm_path = "crate::builtins::math::trigonometry::tan"
)]
fn tan_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let template = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => tan_gpu(handle)?,
        Value::Complex(re, im) => {
            let (out_re, out_im) = tan_complex_components(re, im);
            Value::Complex(out_re, out_im)
        }
        Value::ComplexTensor(ct) => tan_complex_tensor(ct)?,
        Value::CharArray(ca) => tan_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err("tan: expected numeric input".to_string())
        }
        other => tan_real(other)?,
    };
    apply_output_template(base, &template)
}

fn tan_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_tan(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    tan_tensor(tensor).map(tensor::tensor_into_value)
}

fn tan_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("tan", value)?;
    tan_tensor(tensor).map(tensor::tensor_into_value)
}

fn tan_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.tan()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("tan: {e}"))
}

fn tan_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| tan_complex_components(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("tan: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn tan_char_array(array: CharArray) -> Result<Value, String> {
    let data = array
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tan())
        .collect::<Vec<_>>();
    let tensor =
        Tensor::new(data, vec![array.rows, array.cols]).map_err(|e| format!("tan: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn tan_complex_components(re: f64, im: f64) -> (f64, f64) {
    let two_re = 2.0 * re;
    let two_im = 2.0 * im;
    let denom = two_re.cos() + two_im.cosh();
    let real = two_re.sin() / denom;
    let imag = two_im.sinh() / denom;
    (real, imag)
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
                Err("tan: expected prototype after 'like'".to_string())
            } else {
                Err("tan: unrecognised argument for tan".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("tan: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("tan: too many input arguments".to_string()),
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
                Err("tan: complex prototypes for 'like' are not supported yet".to_string())
            }
            _ => Err(
                "tan: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "tan: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("tan: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("tan: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("tan: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "tan: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("tan: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, StringArray, Tensor};

    #[test]
    fn tan_scalar_pi_over_four() {
        let result = tan_builtin(Value::Num(std::f64::consts::FRAC_PI_4), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tan_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::FRAC_PI_4], vec![2, 1]).unwrap();
        let result = tan_builtin(Value::Tensor(tensor), Vec::new()).expect("tan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert!((out.data[0] - 0.0).abs() < 1e-12);
                assert!((out.data[1] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tan_string_input_errors() {
        let err = tan_builtin(Value::from("invalid"), Vec::new()).expect_err("expected error");
        assert!(err.contains("numeric"));
    }

    #[test]
    fn tan_int_promotes() {
        let result = tan_builtin(Value::Int(IntValue::I32(1)), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1f64.tan()).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn tan_complex_scalar_matches_formula() {
        let result = tan_builtin(Value::Complex(1.0, 0.5), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = tan_complex_components(1.0, 0.5);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tan_complex_on_real_axis_matches_real_value() {
        let angle = std::f64::consts::FRAC_PI_2 * 0.9;
        let result = tan_builtin(Value::Complex(angle, 0.0), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - angle.tan()).abs() < 1e-12);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tan_char_array_roundtrip() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = tan_builtin(Value::CharArray(chars), Vec::new()).expect("tan");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = ['A', 'B']
                    .iter()
                    .map(|&ch| (ch as u32 as f64).tan())
                    .collect();
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tan_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.2, -0.3, 1.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(Value::GpuTensor(handle), Vec::new()).expect("tan");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[test]
    fn tan_like_missing_prototype_errors() {
        let err =
            tan_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[test]
    fn tan_like_complex_prototype_errors() {
        let err = tan_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("complex prototypes"));
    }

    #[test]
    fn tan_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.3, 0.6], vec![3, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = tan_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("tan");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn tan_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("tan");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[test]
    fn tan_like_rejects_extra_arguments() {
        let err = tan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[test]
    fn tan_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 0.1], vec![2, 1]).unwrap();
        let result = tan_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> = tensor.data.iter().map(|&v| v.tan()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn tan_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tan_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tan_unrecognised_option_errors() {
        let err =
            tan_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        assert!(err.contains("unrecognised argument"));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn tan_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 0.25, -0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = tan_real(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = tan_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
                }
            }
            _ => panic!("unexpected comparison result"),
        }
    }
}
