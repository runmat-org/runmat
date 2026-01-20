//! MATLAB-compatible `zeros` builtin with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

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
        name = "zeros",
        builtin_path = "crate::builtins::array::creation::zeros"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "zeros"
category: "array/creation"
keywords: ["zeros", "array", "logical", "gpu", "like"]
summary: "Create arrays filled with zeros within the MATLAB language."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider zero-allocation hooks when available; otherwise falls back to uploading a host tensor."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::zeros::tests"
  integration: "builtins::array::zeros::tests::zeros_gpu_like_alloc"
---

# What does the `zeros` function do in MATLAB / RunMat?
`zeros` creates arrays filled with zeros. RunMat mirrors MATLAB semantics across the scalar,
vector, matrix, and N-D forms, including `'like'` and `'logical'` options.

## How does the `zeros` function behave in MATLAB / RunMat?
- `zeros()` returns the scalar `0`.
- `zeros(n)` returns an `n × n` double array.
- `zeros(m, n, ...)` returns a dense double array with the requested dimensions.
- `zeros(sz)` accepts a size vector (row or column) and returns an array with `prod(sz)` elements arranged using MATLAB column-major ordering.
- `zeros(A)` returns an array of zeros with the same size (and device residency) as `A`.
- `zeros(___, 'logical')` returns a logical array instead of double precision.
- `zeros(___, 'like', prototype)` matches the device residency and numeric/logical flavour of `prototype`.

## `zeros` Function GPU Execution Behaviour
When the prototype or `'like'` argument is a GPU tensor, RunMat asks the active acceleration
provider to allocate a zero-filled buffer in-place. Providers that do not yet support the zero
allocation hooks automatically fall back to uploading a host zero tensor, guaranteeing correct
behaviour at the cost of an extra transfer.

## Examples of using the `zeros` function in MATLAB / RunMat

### Creating a 2x3 matrix of zeros

```matlab
A = zeros(2, 3);
```

Expected output:

```matlab
A = [0 0 0; 0 0 0];
```

### Creating a 4x1 logical matrix of zeros

```matlab
mask = zeros(4, 1, 'logical');
```
Expected output:

```matlab
mask = [0 0 0 0];
```

### Creating a 128x128 matrix of zeros on a GPU

In RunMat:

```
H = zeros(128, 128);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
H = gpuArray(zeros(128, 128));

% OR:

G = gpuArray(rand(128, 128));
H = zeros(128, 128, 'like', G);
```

In both cases, the expected output is:

```matlab
H = [0 0 0 ... 0; 0 0 0 ... 0; ...; 0 0 0 ... 0];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `zeros` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `zeros` function?

Use `zeros` whenever you need to create arrays pre-filled with the value `0`, such as for initializing weights, creating masks, or testing other algorithms. Preallocating with `zeros` ensures correct type and shape throughout your code and helps prevent bugs caused by uninitialized arrays.

### Does `zeros` produce double arrays by default?

Yes, by default, `zeros` creates dense double-precision arrays unless you explicitly specify a type such as `'logical'` or use the `'like'` argument to match a prototype array.

### What does `zeros(n)` return?

`zeros(n)` returns an `n × n` dense double-precision matrix filled with zeros. For example, `zeros(3)` yields a 3-by-3 matrix of all zeros.

### How do I create a logical array of zeros?

Pass `'logical'` as the last argument:
```matlab
mask = zeros(5, 1, 'logical');
```
This produces a 5x1 logical array, i.e., all elements have value `false` (`0` in binary).

### How do I match the type and device residency of an existing array?

Use the `'like', prototype` syntax:
```matlab
A = gpuArray(rand(2,2));
B = zeros(2, 2, 'like', A);
```
`B` will be a GPU array with the same type and shape as `A`.

### Can I create N-dimensional arrays with zeros?

Yes! Pass more than two dimension arguments (or a size vector):
```matlab
T = zeros(2, 3, 4);
```
This creates a 2×3×4 tensor of zeros.

### How does `zeros(A)` behave?

If you call `zeros(A)`, where `A` is an array, the result is a new array of zeros with the same shape as `A`.

### Is the output always dense?

Yes. `zeros` always produces a dense array. For sparse matrices of zeros, use `sparse` with appropriate arguments.

### What if I call `zeros` with no arguments?

`zeros()` returns the scalar value `0`.

### Can I use `zeros` to preallocate arrays for later assignment?

Absolutely. Preallocating with `zeros` (or `ones`) and then filling in values is a recommended practice for efficiency and code clarity when the final values are known to overwrite the initial zeros.

## See Also
[ones](./ones), [eye](./eye), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `zeros` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/zeros.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/zeros.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zeros",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("zeros"),
        ProviderHook::Custom("zeros_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device zeros when providers expose dedicated hooks; otherwise falls back to host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("zeros").build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zeros",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                ScalarType::I32 => "0".to_string(),
                ScalarType::Bool => "false".to_string(),
            };
            Ok(zero)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises zeros as literal constants; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "zeros",
    category = "array/creation",
    summary = "Create arrays filled with zeros.",
    keywords = "zeros,array,logical,gpu,like",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::creation::zeros"
)]
fn zeros_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedZeros::parse(rest)?;
    build_output(parsed)
}

struct ParsedZeros {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// Single-precision request. Host tensors are stored as f64 today; we
    /// treat 'single' as a request for a numeric zeros tensor and honour
    /// single precision when allocating on GPU via 'like' or provider hooks.
    Single,
    Logical,
    Like(Value),
}

impl ParsedZeros {
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
                                "zeros: multiple 'like' specifications are not supported",
                            ));
                        }
                        if class_override.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with other class specifiers",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("zeros: expected prototype after 'like'"));
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
                                "zeros: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with 'double'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "zeros: cannot combine 'like' with 'single'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "zeros: unrecognised option '{other}'"
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

fn build_output(parsed: ParsedZeros) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => zeros_double(&parsed.shape),
        OutputTemplate::Single => zeros_single(&parsed.shape),
        OutputTemplate::Logical => zeros_logical(&parsed.shape),
        OutputTemplate::Like(proto) => zeros_like(&proto, &parsed.shape),
    }
}

fn zeros_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

fn zeros_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec())))
}

fn zeros_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => zeros_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let tensor = ComplexTensor::zeros(shape.to_vec());
            Ok(Value::ComplexTensor(tensor))
        }
        Value::GpuTensor(handle) => zeros_like_gpu(handle, shape),
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => zeros_single(shape),
            NumericDType::F64 => zeros_double(shape),
        },
        Value::Num(_) | Value::Int(_) => zeros_double(shape),
        Value::CharArray(_) | Value::Cell(_) => zeros_double(shape),
        _ => zeros_double(shape),
    }
}

fn zeros_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        let attempt = if handle.shape == shape {
            provider.zeros_like(handle)
        } else {
            provider.zeros(shape)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "provider-like-error");
        }
        // Fallback: build a host tensor with dtype matching provider precision and upload
        let host = tensor::zeros_with_dtype(shape, dtype)?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_zeros_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("zeros: {e}"))?;
    log_zeros_fallback(shape, NumericDType::F32, "gather-fallback");
    zeros_like(&gathered, shape)
}

fn zeros_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_zeros_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
    };
    if provider.precision() != precision {
        log_zeros_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.zeros(shape) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!("zeros: provider zeros failed ({err}); falling back to host tensor path");
            log_zeros_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn zeros_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_ZEROS_FALLBACK"),
            Ok(value)
                if value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_zeros_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !zeros_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[zeros_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
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
                    "zeros: matrix dimensions must be non-negative",
                ));
            }
            Ok(Some(vec![dim as usize]))
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| Some(vec![d])),
        Value::Tensor(t) => dims_from_tensor(t).map_err(builtin_error),
        Value::LogicalArray(l) => dims_from_logical(l).map_err(builtin_error),
        _ => Ok(None),
    }
}

fn parse_numeric_dimension(n: f64) -> crate::BuiltinResult<usize> {
    if !n.is_finite() {
        return Err(builtin_error("zeros: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(builtin_error(
            "zeros: matrix dimensions must be non-negative",
        ));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(builtin_error("zeros: dimensions must be integers"));
    }
    Ok(rounded as usize)
}

fn dims_from_tensor(tensor: &Tensor) -> Result<Option<Vec<usize>>, String> {
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

fn dims_from_logical(logical: &LogicalArray) -> Result<Option<Vec<usize>>, String> {
    let _ = logical;
    Ok(None)
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("zeros: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_default_scalar() {
        let result = zeros_builtin(Vec::new()).expect("zeros");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data.len(), 8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_tensor_infers_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| re == 0.0 && im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_uses_shape_argument_when_combined_with_like() {
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(shape_source.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_without_explicit_shape_uses_prototype_shape() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = zeros_builtin(args).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_empty_input_returns_empty_matrix() {
        let empty = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = zeros_builtin(vec![Value::Tensor(empty)]).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_conflicting_like_and_logical_is_error() {
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(zeros_builtin(args).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_like_alloc() {
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
            let result = zeros_builtin(args).expect("zeros");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 0.0));
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
    fn zeros_wgpu_single_allocates_gpu_without_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = zeros_single(&[2, 2]).expect("zeros single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered.data.iter().all(|&x| x == 0.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
