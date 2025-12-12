//! MATLAB-compatible `reshape` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_numel;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "reshape",
        builtin_path = "crate::builtins::array::shape::reshape"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "reshape"
category: "array/shape"
keywords: ["reshape", "resize", "dimensions", "gpu", "auto dimension"]
summary: "Rearrange the dimensions of an array without changing its data."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Providers update GPU tensor shape metadata in place; no kernels are dispatched."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::reshape::tests"
  integration: "builtins::array::shape::reshape::tests::reshape_gpu_preserves_handle_shape"
---

# What does the `reshape` function do in MATLAB / RunMat?
`reshape(A, newSize)` returns the elements of `A` with a different dimensional layout while
preserving column-major ordering. The total number of elements must remain unchanged.

## How does the `reshape` function behave in MATLAB / RunMat?
- Accepts either a size vector `reshape(A, [m n …])` or individual dimensions `reshape(A, m, n, …)`.
- Exactly one dimension may be specified as `[]`; RunMat infers its value from `numel(A)`.
- Dimensions must be nonnegative integers. Zero-sized dimensions are allowed when `numel(A) == 0`.
- Works on numeric, logical, complex, string, char, GPU, and cell arrays (cell/char currently support up to 2-D).
- Reshaping never copies data; it only reinterprets layout metadata.
- Scalar inputs follow MATLAB semantics: `reshape(5, 1, 1)` yields the scalar `5`, while larger shapes return dense arrays.

## `reshape` Function GPU Execution Behavior
When the input lives on the GPU, RunMat asks the active acceleration provider to apply the `reshape`
hook so the backend can update its residency metadata. No data transfers or kernel launches are needed,
so `gpuArray` inputs stay on the device. Providers that do not override the hook fall back to updating
the tensor handle directly, which is sufficient for the in-process reference backend.

## Examples of using the `reshape` function in MATLAB / RunMat

### Reshaping a row vector into a matrix
```matlab
A = 1:12;
B = reshape(A, [3, 4]);
```
Expected output:
```matlab
B =
    1     4     7    10
    2     5     8    11
    3     6     9    12
```

### Using an automatically inferred dimension
```matlab
A = 1:18;
B = reshape(A, 3, []);
```
Expected output:
```matlab
size(B)  % => [3 6]
```

### Reshaping into three dimensions
```matlab
A = 1:24;
C = reshape(A, [2, 3, 4]);
```
Expected output:
```matlab
size(C)  % => [2 3 4]
```

### Reshaping logical arrays preserves type
```matlab
mask = logical([1 0 1 0 1 0]);
grid = reshape(mask, 2, 3);
```
Expected output:
```matlab
grid =
     1     1     1
     0     0     0
```

### Reshaping GPU data without gathering
```matlab
G = gpuArray(1:1000);
H = reshape(G, 10, 100);
```
Expected behavior: `H` stays on the GPU with shape `[10 100]`, and no host copies occur.

### Handling zero-sized dimensions
```matlab
E = reshape([], 0, 3);
```
Expected output:
```matlab
size(E)  % => [0 3]
```

## FAQ

**Can I reshape using a size vector or separate arguments?**  
Yes. `reshape(A, [m n p])` and `reshape(A, m, n, p)` are equivalent.

**How many automatic (`[]`) dimensions are allowed?**  
At most one. RunMat reports `reshape: can only specify a single [] dimension` when more are provided.

**Does `reshape` copy data?**  
No. It only repackages metadata, so large arrays—including GPU arrays—are reshaped instantly.

**Can I reshape empty arrays?**  
Yes. You can reshape to any layout whose product is zero. Automatic dimensions become zero when `numel(A) == 0`.

**Why do char and cell arrays reject 3-D reshapes?**  
The current runtime stores char and cell arrays as 2-D matrices. Future releases will lift this restriction.

**Does reshape change data ordering?**  
No. RunMat uses MATLAB's column-major ordering, so the memory layout is preserved exactly.

**What happens if the dimension product mismatches `numel(A)`?**  
RunMat raises `reshape: product of dimensions (X) must equal numel(A) (Y)`.

**Is reshape compatible with complex inputs?**  
Yes. Complex scalars become complex tensors when the target shape has more than one element.

## See Also
- [`size`](./size)
- [`ndims`](./ndims)
- [`numel`](./numel)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)

## Source & Feedback
- Full implementation: `crates/runmat-runtime/src/builtins/array/shape/reshape.rs`
- Found a bug or behavioral difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose).
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "reshape",
    op_kind: GpuOpKind::Custom("reshape"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("reshape")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers update residency metadata via custom reshape hook; no kernel launches required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "reshape",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Reshape influences fusion layout but emits no kernels; fusion planner treats it as a metadata op.",
};

#[runtime_builtin(
    name = "reshape",
    category = "array/shape",
    summary = "Rearrange the dimensions of an array without changing its data.",
    keywords = "reshape,resize,dimensions,gpu,auto",
    accel = "shape",
    builtin_path = "crate::builtins::array::shape::reshape"
)]
fn reshape_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Err("reshape: size information missing".to_string());
    }
    let tokens = parse_size_arguments(&rest)?;
    let numel = value_numel(&value);
    let dims = finalize_dimensions(tokens, numel)?;
    reshape_value(value, &dims)
}

fn reshape_value(value: Value, dims: &[usize]) -> Result<Value, String> {
    match value {
        Value::Tensor(tensor) => {
            let Tensor { data, .. } = tensor;
            Tensor::new(data, dims.to_vec())
                .map(Value::Tensor)
                .map_err(|e| format!("reshape: {e}"))
        }
        Value::ComplexTensor(ct) => {
            let ComplexTensor { data, .. } = ct;
            ComplexTensor::new(data, dims.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| format!("reshape: {e}"))
        }
        Value::LogicalArray(logical) => {
            let LogicalArray { data, .. } = logical;
            LogicalArray::new(data, dims.to_vec())
                .map(Value::LogicalArray)
                .map_err(|e| format!("reshape: {e}"))
        }
        Value::String(s) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::String(s))
            } else {
                StringArray::new(vec![s], dims.to_vec())
                    .map(Value::StringArray)
                    .map_err(|e| format!("reshape: {e}"))
            }
        }
        Value::StringArray(strings) => {
            let StringArray { data, .. } = strings;
            StringArray::new(data, dims.to_vec())
                .map(Value::StringArray)
                .map_err(|e| format!("reshape: {e}"))
        }
        Value::CharArray(chars) => reshape_char_array(chars, dims),
        Value::Cell(cell) => reshape_cell_array(cell, dims),
        Value::GpuTensor(handle) => reshape_gpu_tensor(handle, dims),
        Value::Num(n) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Num(n))
            } else {
                Tensor::new(vec![n], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| format!("reshape: {e}"))
            }
        }
        Value::Int(i) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Int(i))
            } else {
                Tensor::new(vec![i.to_f64()], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| format!("reshape: {e}"))
            }
        }
        Value::Bool(b) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Bool(b))
            } else {
                let fill = if b { 1u8 } else { 0u8 };
                let total: usize = dims.iter().product();
                LogicalArray::new(vec![fill; total], dims.to_vec())
                    .map(Value::LogicalArray)
                    .map_err(|e| format!("reshape: {e}"))
            }
        }
        Value::Complex(re, im) => reshape_complex_scalar(re, im, dims),
        other => Err(format!(
            "reshape: unsupported input type {:?}; expected numeric, logical, char, string, cell, or gpu array",
            other
        )),
    }
}

fn reshape_complex_scalar(re: f64, im: f64, dims: &[usize]) -> Result<Value, String> {
    let total: usize = dims.iter().copied().product();
    if total != 1 {
        return Err(format!(
            "reshape: product of dimensions ({total}) must equal numel(A) (1)"
        ));
    }

    if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
        Ok(Value::Complex(re, im))
    } else {
        ComplexTensor::new(vec![(re, im)], dims.to_vec())
            .map(Value::ComplexTensor)
            .map_err(|e| format!("reshape: {e}"))
    }
}

fn reshape_char_array(ca: CharArray, dims: &[usize]) -> Result<Value, String> {
    let (rows, cols) = match dims.len() {
        0 => return Err("reshape: size vector must contain at least one element".to_string()),
        1 => (dims[0], 1),
        2 => (dims[0], dims[1]),
        _ => {
            return Err("reshape: char arrays currently support at most two dimensions".to_string())
        }
    };
    CharArray::new(ca.data, rows, cols)
        .map(Value::CharArray)
        .map_err(|e| format!("reshape: {e}"))
}

fn reshape_cell_array(ca: CellArray, dims: &[usize]) -> Result<Value, String> {
    let (rows, cols) = match dims.len() {
        0 => return Err("reshape: size vector must contain at least one element".to_string()),
        1 => (dims[0], 1),
        2 => (dims[0], dims[1]),
        _ => {
            return Err("reshape: cell arrays currently support at most two dimensions".to_string())
        }
    };
    CellArray::new_handles(ca.data, rows, cols)
        .map(Value::Cell)
        .map_err(|e| format!("reshape: {e}"))
}

fn reshape_gpu_tensor(
    handle: runmat_accelerate_api::GpuTensorHandle,
    dims: &[usize],
) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        provider
            .reshape(&handle, dims)
            .map(Value::GpuTensor)
            .map_err(|e| format!("reshape: {e}"))
    } else {
        let mut updated = handle;
        updated.shape = dims.to_vec();
        Ok(Value::GpuTensor(updated))
    }
}

#[derive(Clone, Copy, Debug)]
enum DimToken {
    Known(usize),
    Auto,
}

fn parse_size_arguments(args: &[Value]) -> Result<Vec<DimToken>, String> {
    if args.len() == 1 {
        match &args[0] {
            Value::Tensor(t) => parse_size_vector(t),
            Value::Int(_) | Value::Num(_) | Value::Bool(_) => {
                Ok(vec![parse_size_scalar(&args[0])?])
            }
            Value::GpuTensor(_) => Err(
                "reshape: size vector must be numeric; gpu tensors are not supported as size arguments"
                    .to_string(),
            ),
            Value::LogicalArray(la) => {
                if la.data.is_empty() {
                    Err("reshape: size vector must contain at least one element".to_string())
                } else {
                    let tensor = tensor::logical_to_tensor(la)
                        .map_err(|e| format!("reshape: failed to parse size vector: {e}"))?;
                    parse_size_vector(&tensor)
                }
            }
            other => Err(format!(
                "reshape: size vector must be numeric, got {:?}",
                other
            )),
        }
    } else {
        args.iter().map(parse_size_scalar).collect()
    }
}

fn parse_size_vector(t: &Tensor) -> Result<Vec<DimToken>, String> {
    if !is_vector(t) {
        return Err("reshape: size vector must be a row or column vector".to_string());
    }
    if t.data.is_empty() {
        return Err("reshape: size vector must contain at least one element".to_string());
    }
    t.data.iter().map(|&v| parse_dim_value(v)).collect()
}

fn parse_size_scalar(value: &Value) -> Result<DimToken, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err("reshape: size arguments must be nonnegative integers".to_string());
            }
            Ok(DimToken::Known(raw as usize))
        }
        Value::Num(n) => parse_num_scalar(*n).map(DimToken::Known),
        Value::Bool(b) => Ok(DimToken::Known(if *b { 1 } else { 0 })),
        Value::Tensor(t) => {
            if t.data.is_empty() {
                Ok(DimToken::Auto)
            } else if t.data.len() == 1 {
                parse_dim_value(t.data[0])
            } else {
                Err("reshape: size arguments must be scalars".to_string())
            }
        }
        Value::LogicalArray(la) => {
            if la.data.is_empty() {
                Ok(DimToken::Auto)
            } else if la.data.len() == 1 {
                Ok(DimToken::Known(if la.data[0] != 0 { 1 } else { 0 }))
            } else {
                Err("reshape: size arguments must be scalars".to_string())
            }
        }
        Value::GpuTensor(_) => Err(
            "reshape: size arguments must be numeric scalars; gpu tensors are not supported"
                .to_string(),
        ),
        other => Err(format!(
            "reshape: size arguments must be numeric scalars, got {:?}",
            other
        )),
    }
}

fn parse_dim_value(raw: f64) -> Result<DimToken, String> {
    if !raw.is_finite() {
        return Err("reshape: size arguments must be finite".to_string());
    }
    if raw == 0.0 {
        return Ok(DimToken::Known(0));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("reshape: size arguments must be integers".to_string());
    }
    if rounded < 0.0 {
        return Err("reshape: size arguments must be nonnegative integers".to_string());
    }
    if rounded > (usize::MAX as f64) {
        return Err("reshape: size argument is too large".to_string());
    }
    Ok(DimToken::Known(rounded as usize))
}

fn parse_num_scalar(raw: f64) -> Result<usize, String> {
    if !raw.is_finite() {
        return Err("reshape: size arguments must be finite".to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("reshape: size arguments must be integers".to_string());
    }
    if rounded < 0.0 {
        return Err("reshape: size arguments must be nonnegative integers".to_string());
    }
    if rounded > (usize::MAX as f64) {
        return Err("reshape: size argument is too large".to_string());
    }
    Ok(rounded as usize)
}

fn finalize_dimensions(tokens: Vec<DimToken>, numel: usize) -> Result<Vec<usize>, String> {
    if tokens.is_empty() {
        return Err("reshape: size vector must contain at least one element".to_string());
    }

    let mut dims = Vec::with_capacity(tokens.len());
    let mut known_product: usize = 1;
    let mut auto_index: Option<usize> = None;

    for (idx, token) in tokens.iter().enumerate() {
        match token {
            DimToken::Known(value) => {
                if *value == 0 {
                    known_product = 0;
                } else if known_product != 0 {
                    known_product = known_product.checked_mul(*value).ok_or_else(|| {
                        "reshape: product of dimensions exceeds usize range".to_string()
                    })?;
                }
                dims.push(*value);
            }
            DimToken::Auto => {
                if auto_index.is_some() {
                    return Err("reshape: can only specify a single [] dimension".to_string());
                }
                auto_index = Some(idx);
                dims.push(1); // placeholder
            }
        }
    }

    if let Some(auto) = auto_index {
        if known_product == 0 {
            if numel != 0 {
                return Err(format!(
                    "reshape: product of dimensions (0) must equal numel(A) ({numel})"
                ));
            }
            dims[auto] = 0;
        } else if !numel.is_multiple_of(known_product) {
            return Err(format!(
                "reshape: product of dimensions ({}) must equal numel(A) ({numel})",
                known_product
            ));
        } else {
            dims[auto] = numel / known_product;
        }
    } else if known_product != numel {
        return Err(format!(
            "reshape: product of dimensions ({known_product}) must equal numel(A) ({numel})"
        ));
    }

    Ok(dims)
}

fn is_vector(t: &Tensor) -> bool {
    t.shape.iter().filter(|&&dim| dim > 1).count() <= 1
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    fn tensor_from_slice(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[test]
    fn reshape_vector_to_matrix() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(tensor_from_slice(&[3.0, 4.0], &[1, 2]))],
        )
        .expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4]);
                assert_eq!(out.data, (1..=12).map(|v| v as f64).collect::<Vec<_>>());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn reshape_with_auto_dimension() {
        let data: Vec<f64> = (1..=18).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[18, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::from(3.0), Value::Tensor(empty)];
        let result = reshape_builtin(Value::Tensor(tensor), args).expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 6]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn reshape_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], vec![6, 1]).expect("logical");
        let result = reshape_builtin(
            Value::LogicalArray(logical),
            vec![Value::from(2.0), Value::from(3.0)],
        )
        .expect("reshape");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn reshape_char_array_single_dimension_becomes_column() {
        let chars = CharArray::new("abcd".chars().collect(), 1, 4).expect("char array");
        let result =
            reshape_builtin(Value::CharArray(chars), vec![Value::from(4.0)]).expect("reshape");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 4);
                assert_eq!(out.cols, 1);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "abcd");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[test]
    fn reshape_cell_array_two_dimensional() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell");
        let result = reshape_builtin(Value::Cell(cell), vec![Value::from(2.0), Value::from(1.0)])
            .expect("reshape");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                let first = out.get(0, 0).expect("first cell");
                let second = out.get(1, 0).expect("second cell");
                assert!(matches!(first, Value::Num(f) if (f - 1.0).abs() < 1e-12));
                assert!(matches!(second, Value::Num(f) if (f - 2.0).abs() < 1e-12));
            }
            other => panic!("expected CellArray, got {other:?}"),
        }
    }

    #[test]
    fn reshape_string_scalar_high_rank() {
        let result = reshape_builtin(
            Value::String("runmat".to_string()),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1, 1]);
                assert_eq!(sa.data, vec!["runmat".to_string()]);
            }
            other => panic!("expected StringArray, got {other:?}"),
        }
    }

    #[test]
    fn reshape_gpu_preserves_handle_shape() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
            let tensor = tensor_from_slice(&data, &[3, 4]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(2.0), Value::from(6.0)],
            )
            .expect("reshape");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape, vec![2, 6]);
                    assert_eq!(out.buffer_id, handle.buffer_id);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_updates_provider_shape() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[3, 4]);
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(2.0), Value::from(6.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.shape, vec![2, 6]);
        let host = provider.download(&reshaped).expect("download");
        assert_eq!(host.shape, vec![2, 6]);
        assert_eq!(host.data, tensor.data);
    }

    #[test]
    fn reshape_mismatched_elements_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(4.0), Value::from(4.0)],
        )
        .expect_err("should fail");
        assert!(err.contains("product of dimensions"));
    }

    #[test]
    fn reshape_multiple_auto_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty.clone()), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(err.contains("single []"));
    }

    #[test]
    fn reshape_accepts_zero_sized_dimension() {
        let tensor = tensor_from_slice(&[], &[0, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(0.0), Value::from(3.0)],
        )
        .expect("reshape zero");
        match result {
            Value::Tensor(out) => assert_eq!(out.shape, vec![0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn reshape_int_scalar_to_vector() {
        let value = Value::Int(IntValue::I32(5));
        let err = reshape_builtin(value.clone(), vec![Value::from(1.0), Value::from(5.0)])
            .expect_err("should fail because numel mismatch");
        assert!(err.contains("numel"));
        let ok = reshape_builtin(value, vec![Value::from(1.0), Value::from(1.0)])
            .expect("reshape scalar");
        assert!(matches!(ok, Value::Int(_)));
    }

    #[test]
    fn reshape_complex_scalar_high_rank() {
        let result = reshape_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 1, 1]);
                assert_eq!(ct.data, vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn reshape_auto_dimension_mismatch_reports_product() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(5.0), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(
            err.contains("5"),
            "expected product to appear in error message, got {err}"
        );
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
