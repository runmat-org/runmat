//! MATLAB-compatible `double` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    random_args::keyword_of,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
        FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
        ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "double"
category: "math/elementwise"
keywords: ["double", "float64", "cast", "convert to double", "gpuArray double"]
summary: "Convert numeric values, logical masks, characters, and gpuArray handles to double precision."
references:
  - https://www.mathworks.com/help/matlab/ref/double.html
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f64"]
  broadcasting: "matlab"
  notes: "Prefers the provider's `unary_double` hook on float64-capable backends; gathers to the host when the GPU cannot represent doubles and only re-uploads when supported."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: wgpu
tested:
  unit: "builtins::math::elementwise::double::tests"
  gpu: "builtins::math::elementwise::double::tests::double_gpu_roundtrip"
  docs: "builtins::math::elementwise::double::tests::doc_examples_present"
  wgpu: "builtins::math::elementwise::double::tests::double_wgpu_matches_cpu"
---

# What does the `double` function do in MATLAB / RunMat?
`double(X)` promotes scalars, arrays, complex values, logical masks, character data, and gpuArray
handles to MATLAB’s default double-precision (`float64`) representation while preserving shapes
and column-major layout.

## How does the `double` function behave in MATLAB / RunMat?
- Numeric inputs that are already double precision are returned unchanged; single-precision and
  integer values are promoted without altering shapes.
- Logical inputs become dense double arrays containing `0` and `1`, matching MATLAB’s promotion
  rules used by arithmetic on logical masks.
- Character arrays are converted to their Unicode code points and returned as doubles (`'A'`
  becomes `65`).
- Complex scalars and arrays remain complex, but both components are stored as double precision.
- Empty arrays stay empty, orientation is preserved, and singleton dimensions are untouched.
- Strings, structs, cells, objects, and other unsupported classes raise MATLAB-style errors of the
  form `"double: conversion to double from <type> is not possible"`.

## `double` Function GPU Execution Behaviour
RunMat first inspects the active acceleration provider:

1. **Provider exposes float64 (double) precision:** gpuArray inputs are kept on device. The runtime
   downloads once only when the provider lacks a native cast primitive, performs the conversion on
   the host, and uploads the double-precision result back to the device, freeing the original
   handle. Providers may later add a `unary_double` hook to avoid the temporary host hop.
2. **Provider is float32-only:** gpuArray inputs are gathered to host memory because the backend
   cannot store double precision. The result is returned as a host tensor; subsequent operations can
   choose to re-upload if profitable.
3. **No provider registered:** gpuArray values behave like gathered host arrays, mirroring MATLAB’s
   behaviour when Parallel Computing Toolbox is absent.

All GPU fallbacks are documented so you know exactly when data leaves the device.

## Examples of using the `double` function in MATLAB / RunMat

### Convert integers to double precision
```matlab
ints = int32([1 2 3]);
doubles = double(ints);
```
Expected output:
```matlab
doubles = [1 2 3];
```

### Promote logical masks for arithmetic
```matlab
mask = logical([0 1 0 1]);
weights = double(mask);
```
Expected output:
```matlab
weights = [0 1 0 1];
```

### Convert character arrays to Unicode code points
```matlab
codes = double('RunMat');
```
Expected output:
```matlab
codes = [82 117 110 77 97 116];
```

### Preserve complex numbers while promoting precision
```matlab
z = [1+2i, 3-4i];
result = double(z);
```
Expected output:
```matlab
result = [1+2i  3-4i];
```

### Convert single-precision gpuArray data to double
```matlab
G = single(gpuArray(1:4));
H = double(G);
gather(H)
```
Expected output (on a float64-capable backend):
```matlab
ans = [1 2 3 4];
```

### Handle double precision with `'like'` prototypes
```matlab
proto = gpuArray.zeros(1, 1, 'double');
out = double([pi 0], 'like', proto);
```
Expected output:
```matlab
out =
  1×2 gpuArray  double
    3.1416         0
```

### Promote matrices without changing shape
```matlab
A = single([1.5 2.25; 3.75 4.5]);
B = double(A);
```
Expected output:
```matlab
B = [1.5 2.25; 3.75 4.5];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat keeps tensors on the GPU whenever the active provider supports double precision. You only
need explicit `gpuArray` / `gather` calls when interfacing with legacy code or when you must force
residency. On float32-only providers, `double` returns host data, matching MATLAB systems where the
GPU lacks native double support.

## FAQ

### Does `double` change values that are already double precision?
No. Existing double data is passed through unchanged; the builtin only promotes other types.

### How are logical inputs handled?
Logical masks become numeric 0/1 doubles, making it easy to apply arithmetic or linear algebra
without extra casts.

### What happens to NaN or Inf values?
IEEE special values survive promotion exactly. NaNs stay NaN, and ±Inf remain ±Inf.

### Can I convert strings with `double`?
No. Strings are not implicitly convertible; convert them to character arrays first with `char` and
then take `double`.

### Will `double` keep results on the GPU?
Yes when the provider supports float64. Otherwise the runtime documents the gather fallback and
returns a host tensor.

### Does `double` allocate new memory?
Yes. Results are materialised in a new tensor or, for scalars, a new numeric value. Fusion may fold
the cast with neighbouring elementwise operations.

### Can I request GPU residency with `'like'`?
Yes. Pass `'like', prototype` to mirror the prototype’s residency. Provide a gpuArray prototype to
keep the result on the device when the backend supports float64.

### How does `double` interact with complex inputs?
Complex values keep both components intact; the builtin simply ensures they are stored as double
precision.

## See Also
[single](./single), [real](./real), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/math/elementwise/double.rs`
- Issues & feature requests: [https://github.com/runmat-org/runmat/issues/new/choose](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "double",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_double",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Casts inputs to float64. Providers without native float64 support gather to host; float64-capable providers keep results on device.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "double",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion treats double as an identity when the execution scalar type is already float64.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("double", DOC_MD);

#[runtime_builtin(
    name = "double",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, logical masks, and gpuArray values to double precision.",
    keywords = "double,float64,cast,gpu",
    accel = "unary"
)]
fn double_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let template = parse_output_template(&rest)?;
    let converted = match value {
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::Bool(flag) => Ok(Value::Num(if flag { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor)),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::ComplexTensor(tensor) => Ok(Value::ComplexTensor(tensor)),
        Value::LogicalArray(array) => double_from_logical(array),
        Value::CharArray(chars) => double_from_char_array(chars),
        Value::GpuTensor(handle) => double_from_gpu(handle),
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_) | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
    }?;
    apply_output_template(converted, &template)
}

fn double_from_logical(array: LogicalArray) -> Result<Value, String> {
    let tensor = tensor::logical_to_tensor(&array).map_err(|e| format!("double: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn double_from_char_array(chars: CharArray) -> Result<Value, String> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor =
        Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("double: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn double_from_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle);

    if let Some(provider) = provider {
        if provider.precision() == ProviderPrecision::F64 {
            match provider.unary_double(&handle) {
                Ok(result) => {
                    return Ok(Value::GpuTensor(result));
                }
                Err(err) => {
                    trace!("double: provider unary_double unavailable ({err}); falling back to host conversion");
                }
            }
        } else {
            trace!(
                "double: provider precision {:?} cannot store float64 values; gathering to host",
                provider.precision()
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    if let Some(provider) = provider {
        if provider.precision() == ProviderPrecision::F64 {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            match provider.upload(&view) {
                Ok(new_handle) => return Ok(Value::GpuTensor(new_handle)),
                Err(err) => {
                    trace!("double: provider upload failed after gather ({err})");
                }
            }
        } else {
            trace!(
                "double: provider precision {:?} does not support float64 outputs; returning host tensor",
                provider.precision()
            );
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn conversion_error(type_name: &str) -> String {
    format!(
        "double: conversion to double from {} is not possible",
        type_name
    )
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
                Err("double: expected prototype after 'like'".to_string())
            } else {
                Err("double: unrecognised argument for double".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("double: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("double: too many input arguments".to_string()),
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
                Err("double: complex prototypes for 'like' are not supported yet".to_string())
            }
            _ => Err(
                "double: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "double: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    if provider.precision() != ProviderPrecision::F64 {
        return Err(
            "double: active acceleration provider does not support float64 storage".to_string(),
        );
    }
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("double: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("double: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("double: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "double: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("double: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::IntValue;

    #[test]
    fn double_scalar_num_is_identity() {
        let value = Value::Num(std::f64::consts::PI);
        let result = double_builtin(value, Vec::new()).expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, std::f64::consts::PI),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[test]
    fn double_promotes_integers() {
        let value = Value::Int(IntValue::I32(42));
        let result = double_builtin(value, Vec::new()).expect("double");
        match result {
            Value::Num(n) => assert_eq!(n, 42.0),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[test]
    fn double_logical_array_returns_tensor() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let result = double_builtin(Value::LogicalArray(logical), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn double_char_array_converts_to_codes() {
        let chars = CharArray::new_row("AB");
        let result = double_builtin(Value::CharArray(chars), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 66.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn double_complex_scalar_is_identity() {
        let result = double_builtin(Value::Complex(1.5, -2.5), Vec::new()).expect("double");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.5);
                assert_eq!(im, -2.5);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn double_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
        let result = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("double");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                assert_eq!(t.data, tensor.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn double_rejects_strings() {
        let err = double_builtin(Value::String("hello".into()), Vec::new()).unwrap_err();
        assert_eq!(
            err,
            "double: conversion to double from string is not possible"
        );
    }

    #[test]
    fn double_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(Value::GpuTensor(handle), Vec::new()).expect("double");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[test]
    fn double_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .expect("upload");
            let result = double_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("double");
            match result {
                Value::GpuTensor(h) => {
                    let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                    assert_eq!(gathered.data, tensor.data);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn double_like_host_gathers_gpu_input() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = double_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("double");
            match result {
                Value::Num(n) => assert_eq!(n, 3.0),
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 1]);
                    assert_eq!(t.data, vec![3.0]);
                }
                other => panic!("expected scalar host value, got {other:?}"),
            }
        });
    }

    #[test]
    fn double_like_missing_prototype_errors() {
        let err =
            double_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert!(err.contains("expected prototype"));
    }

    #[test]
    fn double_like_rejects_extra_arguments() {
        let err = double_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn double_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(vec![1.0, 2.5, -3.75, 4.125], vec![2, 2]).unwrap();
        let cpu_value = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = double_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();

        let gathered = test_support::gather(gpu_value.clone()).expect("gather");
        match cpu_value {
            Value::Tensor(ref ct) => {
                assert_eq!(gathered.shape, ct.shape);
                assert_eq!(gathered.data, ct.data);
            }
            Value::Num(n) => {
                assert_eq!(gathered.data, vec![n]);
            }
            other => panic!("unexpected CPU reference value {other:?}"),
        }

        if provider.precision() == ProviderPrecision::F64 {
            assert!(
                matches!(gpu_value, Value::GpuTensor(_)),
                "expected GPU residency under f64 precision"
            );
        } else {
            assert!(
                !matches!(gpu_value, Value::GpuTensor(_)),
                "expected host fallback when f64 unsupported"
            );
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
