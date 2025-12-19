//! MATLAB-compatible `fill` builtin that creates arrays populated with a constant value.
//!
//! The implementation mirrors the modern RunMat builtin blueprint with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{extract_dims, keyword_of, shape_from_value};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fill",
        builtin_path = "crate::builtins::array::creation::fill"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fill"
category: "array/creation"
keywords: ["fill", "constant", "array", "gpu", "like"]
summary: "Create MATLAB arrays filled with a constant value."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider fill hooks for constant buffers; falls back to host upload when unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "uniform"
requires_feature: null
tested:
  unit: "builtins::array::creation::fill::tests"
  integration: "builtins::array::creation::fill::tests::fill_gpu_like_alloc"
---

# What does the `fill` function do in MATLAB / RunMat?
`fill(value, size, ...)` constructs dense MATLAB arrays whose entries are all the supplied `value`.
It aligns with the MATLAB-compatible RunMat array creation suite (`zeros`, `ones`, `eye`, etc.) while
supporting GPU residency, `'like'` prototypes, and logical/complex outputs.

## How does the `fill` function behave in MATLAB / RunMat?
- `fill(v)` returns the scalar `v`.
- `fill(v, n)` returns an `n × n` matrix with every element equal to `v`.
- `fill(v, m, n, ...)` supports arbitrary N-D dimensions.
- `fill(v, sz)` accepts a size vector (row or column).
- `fill(v, A)` uses the shape (and default type) of `A`.
- `fill(v, ___, 'logical')` produces a logical array where each element is `v ~= 0`.
- `fill(v, ___, 'like', prototype)` matches the type, device residency, and shape of `prototype`.
- When a GPU prototype or `'like'` argument is provided, RunMat keeps the result on the device. When
  provider hooks are incomplete, it falls back to uploading a host tensor to guarantee correctness.

## GPU execution behaviour
RunMat asks the active acceleration provider to materialise the constant directly via the dedicated
`fill` hook. Providers that do not supply this capability fall back to a zero buffer plus scalar add
or, as a last resort, upload the fully materialised host tensor. This guarantees MATLAB-compatible
results while still benefiting from GPU residency whenever possible.

## Examples of using the `fill` function in MATLAB / RunMat

### Creating a 3x3 matrix of 2.5s

```matlab
A = fill(2.5, 3);
```

Expected output:

```matlab
A = [2.5 2.5 2.5; 2.5 2.5 2.5; 2.5 2.5 2.5];
```

### Filling a rectangular matrix with -4

```matlab
B = fill(-4, 2, 5);
```

Expected output:

```matlab
B = [-4 -4 -4 -4 -4; -4 -4 -4 -4 -4];
```

### Filling using a size vector

```matlab
sz = [2 3 4];
C = fill(10, sz);
```

Expected output:

```matlab
size(C)  % 2-by-3-by-4
```

### Creating a logical mask with `fill`

```matlab
mask = fill(3, 4, 1, 'logical');
```

Expected output:

```matlab
mask = [1; 1; 1; 1];
```

### Matching an existing array with `'like'`

```matlab
prototype = rand(2, 3, 'like', gpuArray(1));
D = fill(pi, 2, 3, 'like', prototype);
```

Expected output:

```matlab
D is gpuArray
gather(D) = ones(2, 3) * pi;
```

### Filling a complex array

```matlab
E = fill(1 + 2i, 2, 2, 'complex');
```

Expected output:

```matlab
E = [1+2i 1+2i; 1+2i 1+2i];
```

### Using a prototype to infer shape automatically

```matlab
F = rand(4, 2);
G = fill(7, F);
```

Expected output:

```matlab
isequal(size(G), [4 2])  % true
```

### Filling an empty matrix

```matlab
H = fill(0, 0, 5);
```

Expected output:

```matlab
size(H)  % 0-by-5
```

### Keeping GPU residency without explicit `gpuArray`

```matlab
A = rand(512, 512);
J = fill(0.5, size(A), 'like', gpuArray(A));
```

Expected output:

```matlab
isa(J, 'gpuArray')  % true
```

### Filling with boolean semantics

```matlab
K = fill(false, [2 2], 'logical');
```

Expected output:

```matlab
K = [0 0; 0 0];
```

## FAQ

### Which argument order does `fill` support?
The first argument is always the scalar value. Remaining arguments specify dimensions, size vectors,
or options (`'logical'`, `'like'`, `'double'`, `'complex'`).

### Does `fill` accept non-scalar values?
No. The fill value must be numeric, logical, or complex scalar. Use `repmat` when you need to tile an
array.

### How does `fill` behave with complex values?
If you request complex output (explicitly or via `'like'`), the real and imaginary parts are copied
verbatim. Otherwise the imaginary component must be zero.

### How do logical arrays handle nonzero values?
Logical outputs treat any nonzero real or complex magnitude as `true`, matching MATLAB semantics.

### Can I match GPU residency automatically?
Yes. Pass a GPU prototype via `'like'` and `fill` will allocate on the device. It falls back to host
allocation and upload when the provider cannot construct the constant directly.

### What happens when I omit dimensions?
If you only pass the value, `fill` returns a scalar. When you pass a prototype (e.g., `fill(v, A)`),
the prototype shape is reused. Otherwise RunMat follows MATLAB's default of `1 × 1`.

### Are string or char outputs supported?
Not yet. `fill` currently targets numeric, logical, and complex arrays. Use string-manipulation
functions for text.

### How do I create empty outputs?
Include a zero dimension (`fill(value, 0, n)` or `fill(value, [0 n])`). RunMat returns an empty array
of the requested size and type.

### Does `fill` honour `'single'` outputs?
Single precision outputs are not implemented yet. The function returns an error mirroring other array
creation builtins.

### Is the GPU path deterministic?
Yes. Providers either execute the constant fill entirely on the GPU or RunMat uploads a deterministic
host tensor, ensuring identical results.

## See Also
[zeros](./zeros), [ones](./ones), [repmat](../manipulation/repmat), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/array/creation/fill.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/fill.rs)
- Issues or questions: [RunMat GitHub issue tracker](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::fill")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fill",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("fill"),
        ProviderHook::Custom("fill_like"),
    ],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs dedicated constant-fill kernels; falls back to host upload when the provider reports an error.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::fill")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fill",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let constant = ctx
                .constants
                .first()
                .ok_or(crate::builtins::common::spec::FusionError::MissingInput(0))?;
            Ok(constant.to_string())
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner treats fill as a constant generator backed by a uniform parameter.",
};

#[runtime_builtin(
    name = "fill",
    category = "array/creation",
    summary = "Create arrays filled with a constant value.",
    keywords = "fill,constant,array,gpu,like",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::creation::fill"
)]
fn fill_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered_value =
        crate::dispatcher::gather_if_needed(&value).map_err(|e| format!("fill: {e}"))?;
    let scalar = FillScalar::from_value(&gathered_value)?;
    let parsed = ParsedFill::parse(scalar, rest)?;
    build_output(parsed)
}

struct ParsedFill {
    fill: FillScalar,
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Logical,
    Complex,
    Like(Value),
}

#[derive(Clone, Copy)]
enum FillScalar {
    Real(f64),
    Complex(f64, f64),
    Logical(bool),
}

impl FillScalar {
    fn from_value(value: &Value) -> Result<Self, String> {
        match value {
            Value::Num(n) => Ok(FillScalar::Real(*n)),
            Value::Int(i) => Ok(FillScalar::Real(i.to_f64())),
            Value::Bool(b) => Ok(FillScalar::Logical(*b)),
            Value::LogicalArray(logical) => {
                if logical.data.len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Logical(logical.data[0] != 0))
            }
            Value::Tensor(tensor) => {
                if tensor.data.len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Real(tensor.data[0]))
            }
            Value::Complex(re, im) => Ok(FillScalar::Complex(*re, *im)),
            Value::ComplexTensor(tensor) => {
                if tensor.data.len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Complex(tensor.data[0].0, tensor.data[0].1))
            }
            Value::CharArray(ca) => {
                if ca.data.len() != 1 {
                    return Err("fill: fill value must be a scalar".to_string());
                }
                Ok(FillScalar::Real(ca.data[0] as u32 as f64))
            }
            Value::StringArray(sa) if sa.data.len() == 1 && sa.data[0].len() == 1 => {
                let ch = sa.data[0].chars().next().unwrap();
                Ok(FillScalar::Real(ch as u32 as f64))
            }
            Value::String(s) if s.len() == 1 => {
                let ch = s.chars().next().unwrap();
                Ok(FillScalar::Real(ch as u32 as f64))
            }
            other => Err(format!(
                "fill: fill value must be numeric, logical, or complex scalar (got {other:?})"
            )),
        }
    }

    fn as_real(&self) -> Result<f64, String> {
        match self {
            FillScalar::Real(v) => Ok(*v),
            FillScalar::Logical(b) => Ok(if *b { 1.0 } else { 0.0 }),
            FillScalar::Complex(re, im) => {
                if im.abs() > f64::EPSILON {
                    Err("fill: imaginary component must be zero for real outputs".to_string())
                } else {
                    Ok(*re)
                }
            }
        }
    }

    fn as_complex(&self) -> (f64, f64) {
        match self {
            FillScalar::Real(v) => (*v, 0.0),
            FillScalar::Logical(b) => (if *b { 1.0 } else { 0.0 }, 0.0),
            FillScalar::Complex(re, im) => (*re, *im),
        }
    }

    fn as_bool(&self) -> bool {
        match self {
            FillScalar::Real(v) => *v != 0.0,
            FillScalar::Logical(b) => *b,
            FillScalar::Complex(re, im) => *re != 0.0 || *im != 0.0,
        }
    }
}

impl ParsedFill {
    fn parse(fill: FillScalar, args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err("fill: multiple 'like' specifications are not supported"
                                .to_string());
                        }
                        if class_override.is_some() {
                            return Err(
                                "fill: cannot combine 'like' with class specifiers".to_string()
                            );
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("fill: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto, "fill")?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'logical'".to_string());
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'double'".to_string());
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "complex" => {
                        if like_proto.is_some() {
                            return Err("fill: cannot combine 'like' with 'complex'".to_string());
                        }
                        class_override = Some(OutputTemplate::Complex);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "fill: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("fill: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "fill")? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg, "fill")?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        let default_template = match fill {
            FillScalar::Logical(_) => OutputTemplate::Logical,
            FillScalar::Complex(_, _) => OutputTemplate::Complex,
            FillScalar::Real(_) => OutputTemplate::Double,
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            default_template
        };

        Ok(Self {
            fill,
            shape,
            template,
        })
    }
}

fn build_output(parsed: ParsedFill) -> Result<Value, String> {
    match parsed.template {
        OutputTemplate::Double => fill_double(&parsed.fill, &parsed.shape),
        OutputTemplate::Logical => fill_logical(&parsed.fill, &parsed.shape),
        OutputTemplate::Complex => fill_complex(&parsed.fill, &parsed.shape),
        OutputTemplate::Like(proto) => fill_like(&parsed.fill, &parsed.shape, &proto),
    }
}

fn fill_double(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let tensor = make_real_tensor(fill, shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn fill_logical(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let logical = make_logical_array(fill, shape)?;
    Ok(Value::LogicalArray(logical))
}

fn fill_complex(fill: &FillScalar, shape: &[usize]) -> Result<Value, String> {
    let tensor = make_complex_tensor(fill, shape)?;
    Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
}

fn fill_like(fill: &FillScalar, shape: &[usize], proto: &Value) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => fill_logical(fill, shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => fill_complex(fill, shape),
        Value::GpuTensor(handle) => fill_like_gpu(fill, shape, handle),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => fill_double(fill, shape),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_) => {
            Err("fill: character, string, and cell prototypes are not supported yet".to_string())
        }
        other => Err(format!(
            "fill: unsupported prototype type {:?} for 'like' output",
            other
        )),
    }
}

fn fill_like_gpu(
    fill: &FillScalar,
    shape: &[usize],
    prototype: &GpuTensorHandle,
) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if prototype.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let value = fill.as_real()?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if prototype.shape == shape {
            provider.fill_like(prototype, value)
        } else {
            provider.fill(shape, value)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let tensor = make_real_tensor_from_value(value, shape)?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(uploaded) = provider.upload(&view) {
            return Ok(Value::GpuTensor(uploaded));
        }
        return Ok(tensor::tensor_into_value(tensor));
    }

    let tensor = make_real_tensor_from_value(value, shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn make_real_tensor(fill: &FillScalar, shape: &[usize]) -> Result<Tensor, String> {
    let value = fill.as_real()?;
    make_real_tensor_from_value(value, shape)
}

fn make_real_tensor_from_value(value: f64, shape: &[usize]) -> Result<Tensor, String> {
    let count = tensor::element_count(shape);
    let mut data = vec![value; count];
    if count == 0 {
        data.clear();
    }
    Tensor::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

fn make_complex_tensor(fill: &FillScalar, shape: &[usize]) -> Result<ComplexTensor, String> {
    let (re, im) = fill.as_complex();
    let count = tensor::element_count(shape);
    let data = vec![(re, im); count];
    ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

fn make_logical_array(fill: &FillScalar, shape: &[usize]) -> Result<LogicalArray, String> {
    let bit = if fill.as_bool() { 1u8 } else { 0u8 };
    let count = tensor::element_count(shape);
    let mut data = vec![bit; count];
    if count == 0 {
        data.clear();
    }
    LogicalArray::new(data, shape.to_vec()).map_err(|e| format!("fill: {e}"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_scalar_defaults() {
        let result = fill_builtin(Value::Num(5.0), Vec::new()).expect("fill");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = fill_builtin(Value::Num(2.5), args).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&x| (x - 2.5).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rectangular_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = fill_builtin(Value::Num(-4.0), args).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert!(t.data.iter().all(|&x| (x + 4.0).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_size_vector() {
        let sz = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = fill_builtin(Value::Num(10.0), vec![Value::Tensor(sz)]).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 4]);
                assert!(t.data.iter().all(|&x| (x - 10.0).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_logical_option() {
        let args = vec![Value::Num(4.0), Value::from("logical")];
        let result = fill_builtin(Value::Num(3.0), args).expect("fill");
        match result {
            Value::LogicalArray(l) => {
                assert_eq!(l.shape, vec![4, 4]);
                assert!(l.data.iter().all(|&b| b == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_tensor_infers_shape() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = fill_builtin(Value::Num(std::f64::consts::PI), vec![Value::Tensor(proto)])
            .expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t
                    .data
                    .iter()
                    .all(|&x| (x - std::f64::consts::PI).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_complex() {
        let result = fill_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::Num(2.0), Value::from("complex")],
        )
        .expect("fill");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t
                    .data
                    .iter()
                    .all(|&(re, im)| (re - 1.0).abs() < 1e-12 && (im - 2.0).abs() < 1e-12));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rejects_non_scalar_value() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = fill_builtin(Value::Tensor(tensor), Vec::new());
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_requires_real_for_double_output() {
        let result = fill_builtin(
            Value::Complex(1.0, 1.0),
            vec![Value::Num(2.0), Value::Num(2.0), Value::from("double")],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_double_option_generates_real_array() {
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::from("double")];
        let result = fill_builtin(Value::Num(1.5), args).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert!(t.data.iter().all(|&x| (x - 1.5).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_infers_shape_without_dims() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = fill_builtin(Value::Num(4.2), args).expect("fill");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data.iter().all(|&x| (x - 4.2).abs() < 1e-12));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_logical_prototype() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let args = vec![Value::from("like"), Value::LogicalArray(logical)];
        let result = fill_builtin(Value::Num(0.0), args).expect("fill");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![1, 1]);
                assert_eq!(arr.data, vec![0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_rejects_single_precision_option() {
        let result = fill_builtin(
            Value::Num(1.0),
            vec![Value::Num(2.0), Value::from("single")],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_like_logical_conflict_errors() {
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(1.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = fill_builtin(Value::Num(0.0), args);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fill_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = fill_builtin(Value::Num(0.5), args).expect("fill");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| (x - 0.5).abs() < 1e-12));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fill_wgpu_like_matches_cpu() {
        let provider =
            match std::panic::catch_unwind(
                || register_wgpu_provider(WgpuProviderOptions::default()),
            ) {
                Ok(Ok(_)) => {
                    if let Some(p) = runmat_accelerate_api::provider() {
                        p
                    } else {
                        tracing::warn!(
                            "skipping fill_wgpu_like_matches_cpu: provider not registered after init"
                        );
                        return;
                    }
                }
                Ok(Err(err)) => {
                    tracing::warn!(
                        "skipping fill_wgpu_like_matches_cpu: wgpu provider unavailable ({err})"
                    );
                    return;
                }
                Err(_) => {
                    tracing::warn!(
                        "skipping fill_wgpu_like_matches_cpu: wgpu provider initialisation panicked"
                    );
                    return;
                }
            };
        let prototype = provider.fill(&[1, 1], 0.0).expect("prototype allocation");
        let args = vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::GpuTensor(prototype),
        ];
        let target = 0.75;
        let result = fill_builtin(Value::Num(target), args).expect("fill");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.shape, vec![2, 3]);
                for entry in gathered.data {
                    assert!((entry - target).abs() < 1e-12);
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
