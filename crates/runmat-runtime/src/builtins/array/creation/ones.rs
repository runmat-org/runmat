//! MATLAB-compatible `ones` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::NumericDType;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "ones",
        builtin_path = "crate::builtins::array::creation::ones"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ones"
category: "array/creation"
keywords: ["ones", "array", "logical", "gpu", "like"]
summary: "Create arrays filled with ones within the MATLAB language."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider one-allocation hooks when available; otherwise fills via scalar add or uploads from the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::ones::tests"
  integration: "builtins::array::ones::tests::ones_gpu_like_alloc"
---

# What does the `ones` function do in MATLAB / RunMat?
`ones` creates arrays filled with ones. RunMat mirrors MATLAB semantics across the scalar,
vector, matrix, and N-D forms, including `'like'` and `'logical'` options.

## How does the `ones` function behave in MATLAB / RunMat?
- `ones()` returns the scalar `1`.
- `ones(n)` returns an `n × n` double array.
- `ones(m, n, ...)` returns a dense double array with the requested dimensions.
- `ones(sz)` accepts a size vector (row or column) and returns an array with `prod(sz)` elements arranged using MATLAB column-major ordering.
- `ones(A)` returns an array of ones with the same size (and device residency) as `A`.
- `ones(___, 'logical')` returns a logical array instead of double precision (all elements set to `true`).
- `ones(___, 'like', prototype)` matches the device residency and numeric/logical flavour of `prototype`.

## `ones` Function GPU Execution Behaviour
When the prototype or `'like'` argument is a GPU tensor, RunMat asks the active acceleration
provider to allocate a one-filled buffer. Acceleration providers that do not yet support the dedicated hooks
fall back to zero-allocation plus scalar fill or, as a last resort, upload a host tensor.
This guarantees correct behaviour.

## Examples of using the `ones` function in MATLAB / RunMat

### Creating a 3x3 matrix of ones

```matlab
B = ones(3);
```

Expected output:

```matlab
B = [1 1 1; 1 1 1; 1 1 1];
```

### Creating a 4x1 logical matrix of ones

```matlab
mask = ones(4, 1, 'logical');
```

Expected output:

```matlab
mask = [1 1 1 1];
```

### Creating a 64x64 matrix of ones on a GPU

In RunMat:

```
H = ones(64, 64);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
H = gpuArray(ones(64, 64));

% OR:

G = gpuArray(rand(64, 64));
H = ones(64, 64, 'like', G);
```

In both cases, the expected output is:

```matlab
H = [1 1 1 ... 1; 1 1 1 ... 1; ...; 1 1 1 ... 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `ones` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `ones` function?

Use `ones` whenever you need to create arrays pre-filled with the value `1`, such as for initializing weights, creating masks, or testing other algorithms. Preallocating with `ones` ensures correct type and shape throughout your code and helps prevent bugs caused by uninitialized arrays.

### Does `ones` produce double arrays by default?

Yes, by default, `ones` creates dense double-precision arrays unless you explicitly specify a type such as `'logical'` or use the `'like'` argument to match a prototype array.

### What does `ones(n)` return?

`ones(n)` returns an `n × n` dense double-precision matrix filled with ones. For example, `ones(3)` yields a 3-by-3 matrix of all ones.

### How do I create a logical array of ones?

Pass `'logical'` as the last argument:  
```matlab
mask = ones(5, 1, 'logical');
```
This produces a 5x1 logical array, i.e., all elements have value `true` (`1` in binary).

### How do I match the type and device residency of an existing array?

Use the `'like', prototype` syntax:  
```matlab
A = gpuArray(rand(2,2));
B = ones(2, 2, 'like', A);
```
`B` will be a GPU array with the same type and shape as `A`.

### Can I create N-dimensional arrays with ones?

Yes! Pass more than two dimension arguments (or a size vector):  
```matlab
T = ones(2, 3, 4);
```
This creates a 2×3×4 tensor of ones.

### How does `ones(A)` behave?

If you call `ones(A)`, where `A` is an array, the result is a new array of ones with the same shape as `A`.

### Is the output always dense?

Yes. `ones` always produces a dense array. For sparse matrices of ones, use `sparse` with appropriate arguments.

### What if I call `ones` with no arguments?

`ones()` returns the scalar value `1`.

### Can I use `ones` to preallocate arrays for later assignment?

Absolutely. Preallocating with `ones` (or `zeros`) and then filling in values is a recommended practice for efficiency and code clarity when the final values are known to overwrite the initial ones.

## See Also
[zeros](./zeros), [eye](./eye), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `ones` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/ones.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/ones.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::ones")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ones",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("ones"),
        ProviderHook::Custom("ones_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device ones when providers expose dedicated hooks; otherwise falls back to scalar fill or host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("ones").build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::ones")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ones",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                ScalarType::I32 => "1".to_string(),
                ScalarType::Bool => "true".to_string(),
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises ones as inline literals; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "ones",
    category = "array/creation",
    summary = "Create arrays filled with ones.",
    keywords = "ones,array,logical,gpu,like",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::creation::ones"
)]
async fn ones_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedOnes::parse(rest)?;
    build_output(parsed).await
}

struct ParsedOnes {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// See zeros: host tensors are f64; honour 'single' as numeric ones and
    /// allow GPU paths to select f32 where applicable via 'like' or provider hooks.
    Single,
    Logical,
    Like(Value),
}

impl ParsedOnes {
    fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
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
                            return Err(builtin_error(
                                "ones: multiple 'like' specifications are not supported",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(builtin_error(
                                "ones: cannot combine 'like' with other class specifiers",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("ones: expected prototype after 'like'"));
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "ones: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(builtin_error("ones: cannot combine 'like' with 'double'"));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(builtin_error("ones: cannot combine 'like' with 'single'"));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "ones: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg)? {
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
                shape_source = Some(shape_from_value(&arg)?);
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

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedOnes) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => ones_double(&parsed.shape),
        OutputTemplate::Single => ones_single(&parsed.shape),
        OutputTemplate::Logical => ones_logical(&parsed.shape),
        OutputTemplate::Like(proto) => ones_like(&proto, &parsed.shape).await,
    }
}

fn ones_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = tensor::ones(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = tensor::ones_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    LogicalArray::new(vec![1u8; len], shape.to_vec())
        .map(Value::LogicalArray)
        .map_err(|e| builtin_error(format!("ones: {e}")))
}

#[async_recursion::async_recursion(?Send)]
async fn ones_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => ones_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let len = tensor::element_count(shape);
            let data = vec![(1.0, 0.0); len];
            ComplexTensor::new(data, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| builtin_error(format!("ones: {e}")))
        }
        Value::GpuTensor(handle) => ones_like_gpu(handle, shape).await,
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => ones_single(shape),
            NumericDType::F64 => ones_double(shape),
        },
        Value::Num(_) | Value::Int(_) => ones_double(shape),
        Value::CharArray(_) | Value::Cell(_) => ones_double(shape),
        _ => ones_double(shape),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn ones_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.ones_like(handle)
        } else {
            provider.ones(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        if let Ok(zero_handle) = provider.zeros(shape) {
            let add_result = provider.scalar_add(&zero_handle, 1.0);
            let _ = provider.free(&zero_handle);
            if let Ok(filled) = add_result {
                return Ok(Value::GpuTensor(filled));
            }
        }

        if let Ok(host) = tensor::ones_with_dtype(
            shape,
            match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
                runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
            },
        ) {
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            if let Ok(gpu) = provider.upload(&view) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| builtin_error(format!("ones: {e}")))?;
    ones_like(&gathered, shape).await
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

fn extract_dims(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    match value {
        Value::Int(i) => {
            let dim = i.to_i64();
            if dim < 0 {
                return Err(builtin_error(
                    "ones: matrix dimensions must be non-negative",
                ));
            }
            Ok(Some(vec![dim as usize]))
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| Some(vec![d])),
        Value::Tensor(t) => dims_from_tensor(t),
        Value::LogicalArray(l) => dims_from_logical(l),
        _ => Ok(None),
    }
}

fn parse_numeric_dimension(n: f64) -> crate::BuiltinResult<usize> {
    if !n.is_finite() {
        return Err(builtin_error("ones: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(builtin_error(
            "ones: matrix dimensions must be non-negative",
        ));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(builtin_error("ones: dimensions must be integers"));
    }
    Ok(rounded as usize)
}

fn dims_from_tensor(tensor: &Tensor) -> crate::BuiltinResult<Option<Vec<usize>>> {
    let is_row = tensor.rows() == 1;
    let is_column = tensor.cols() == 1;
    let is_scalar = tensor.data.len() == 1;
    if !(is_row || is_column || is_scalar || tensor.shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        match parse_numeric_dimension(v) {
            Ok(dim) => dims.push(dim),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(dims))
}

fn dims_from_logical(_logical: &LogicalArray) -> crate::BuiltinResult<Option<Vec<usize>>> {
    Ok(None)
}

fn shape_from_value(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(builtin_error(format!(
            "ones: unsupported prototype {other:?}"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_default_scalar() {
        let result = block_on(ones_builtin(Vec::new())).expect("ones");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_like_tensor_infers_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| (re, im) == (1.0, 0.0)));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_like_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let args = vec![Value::LogicalArray(logical)];
        let result = block_on(ones_builtin(args)).expect("ones");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!(out.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ones_gpu_like_alloc() {
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
            let result = block_on(ones_builtin(args)).expect("ones");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 1.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
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
    fn ones_wgpu_like_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Build GPU prototype via gpuArray
        let proto = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let g = block_on(crate::call_builtin_async("gpuArray", &[Value::Tensor(proto)]))
            .expect("gpuArray");
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("like"), g];
        let result = block_on(ones_builtin(args)).expect("ones like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ones_wgpu_fusion_with_sin_and_sum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Create ones on GPU (2x2), then sin, then sum along dim=1
        let args = vec![Value::Num(2.0), Value::Num(2.0)];
        let o = block_on(ones_builtin(args)).expect("ones");
        let s = block_on(crate::call_builtin_async("sin", &[o])).expect("sin");
        let summed =
            block_on(crate::call_builtin_async("sum", &[s, Value::Num(1.0)])).expect("sum");
        // Gather and validate shapes; values are deterministic for sin(1)
        let gathered = test_support::gather(summed).expect("gather");
        assert_eq!(gathered.shape, vec![1, 2]);
    }
}
