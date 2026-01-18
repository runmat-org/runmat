//! MATLAB-compatible `repmat` builtin with GPU-aware semantics for RunMat.
//!
//! Replicates arrays by tiling their contents across one or more dimensions.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "repmat",
        builtin_path = "crate::builtins::array::shape::repmat"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "repmat"
category: "array/shape"
keywords: ["repmat", "tile", "replicate", "array", "gpu"]
summary: "Replicate arrays by tiling an input across one or more dimensions."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider tiling hooks when available; otherwise gathers to host memory and re-uploads to preserve GPU residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::repmat::tests"
  integration: "builtins::array::shape::repmat::tests::repmat_gpu_roundtrip"
---

# What does the `repmat` function do in MATLAB / RunMat?
`repmat(A, reps)` tiles the array `A` so that it repeats according to `reps`.
RunMat mirrors MATLAB semantics, supporting scalars, matrices, tensors, logical,
string, char, cell, and GPU-resident arrays.

## How does the `repmat` function behave in MATLAB / RunMat?
- `repmat(A, m, n)` or `repmat(A, [m n])` repeats `A` `m` times along rows and `n`
  times along columns.
- A size vector `[r1 r2 … rN]` can be given as a row or column; trailing dimensions
  default to `1`, letting you replicate only the axes you care about.
- Supplying a scalar replication factor (e.g., `repmat(A, k)`) duplicates `A`
  across every dimension, ensuring the first two dimensions are replicated even
  when the input is a row vector.
- Zero replication factors produce empty dimensions, e.g., `repmat(A, 0, 3)` yields
  an empty array with shape `[0 size(A,2)]`.
- Char and cell arrays follow MATLAB by supporting row/column tiling; additional
  dimensions must be `1`, while numeric, logical, complex, and string arrays
  support full N-D replication.
- Replication factors must be finite, non-negative integers; RunMat raises
  descriptive errors for negative, fractional, or excessively large sizes.
- GPU arrays remain on the device when the acceleration provider implements the
  tiling hook; otherwise, RunMat gathers to host memory, tiles in software, and
  re-uploads the replicated tensor so downstream GPU work keeps residency.

## `repmat` Function GPU Execution Behaviour
RunMat first calls `AccelProvider::repmat`, giving the backend a chance to tile
directly on the device. Providers that have not yet implemented this hook fall
back to a safe path that gathers the tensor, performs tiling on the host, and
uploads the replicated data back to the GPU. The fallback preserves correct
behaviour today while enabling backend authors to drop in optimized kernels as
they become available.

## Examples of using the `repmat` function in MATLAB / RunMat

### Tiling a matrix across rows and columns
```matlab
A = [1 2; 3 4];
B = repmat(A, 2, 3);
```
Expected output:
```matlab
B =
     1     2     1     2     1     2
     3     4     3     4     3     4
     1     2     1     2     1     2
     3     4     3     4     3     4
```

### Using a scalar replication factor
```matlab
row = 1:4;
Tiled = repmat(row, 3);
size(Tiled)
```
Expected output:
```matlab
ans =
     3    12
```

### Replicating into three dimensions
```matlab
A = reshape(1:6, [1 3 2]);
T = repmat(A, [2 1 4]);
size(T)
```
Expected output:
```matlab
ans =
     2     3     8
```

### Repeating logical masks with zero dimensions
```matlab
mask = logical([1 0 1]);
emptyMask = repmat(mask, 0, 3);
size(emptyMask)
```
Expected output:
```matlab
ans =
     0     3
```

### Replicating string scalars into string arrays
```matlab
name = "runmat";
names = repmat(name, 2, 2);
```
Expected output:
```matlab
names =
  2x2 string array
    "runmat"    "runmat"
    "runmat"    "runmat"
```

### Replicating data that lives on the GPU
```matlab
G = gpuArray(magic(3));
T = repmat(G, [2 1]);
result = gather(T);
```
Expected behaviour: `result` is a 6-by-3 matrix that tiles the original `magic(3)`
matrix twice along the row dimension without requiring explicit `gpuArray` calls
in user code.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` directly. RunMat's planner keeps
values on the GPU when it detects that further operations benefit from staying
there. Explicit `gpuArray` calls remain available for compatibility with legacy
MATLAB code and when you want to steer residency manually.

## FAQ

### Are replication factors required to be integers?
Yes. RunMat follows MATLAB and requires every replication factor to be a
non-negative integer. Non-integers raise the descriptive error
`repmat: replication factor <n> must be an integer`.

### What happens when I pass a scalar replication factor?
RunMat duplicates the input along every dimension, ensuring at least the first two
dimensions receive the factor so matrices tile both rows and columns.

### Can I request zero replication along a dimension?
Yes. Zero factors produce empty dimensions while preserving the remaining sizes,
which is useful when constructing placeholder tensors or short-circuiting loops.

### Does `repmat` work with char, string, or cell arrays?
Yes. Char arrays and cell arrays tile across rows and columns (extra dimensions
must currently be `1`), while string arrays, numeric tensors, logical arrays, and
complex tensors support full N-D replication.

### How does `repmat` handle GPU tensors today?
The runtime asks the provider for a device implementation. If none exists (for
example, the in-process test provider), RunMat gathers to the host, tiles there,
and re-uploads the result so downstream GPU kernels still see the replicated tensor.

### Does the result reuse backing storage from the input?
No. `repmat` always creates a new array so modifying the result never mutates the
original input.

### Can replication overflow memory?
RunMat checks for overflow when multiplying shape dimensions. If the requested
size cannot fit in native address space, the builtin raises a descriptive error
before attempting allocation.

### Does `repmat` preserve data ordering?
Yes. Column-major ordering is maintained for numeric, logical, string, and complex
arrays, while char and cell arrays respect their row-major storage conventions
within RunMat.

## See Also
[reshape](./reshape), [permute](./permute), [squeeze](./squeeze), [zeros](./zeros), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/array/shape/repmat.rs`
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose).
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::repmat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "repmat",
    op_kind: GpuOpKind::Custom("tile"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("repmat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to gather + upload when providers lack a native tiling implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::repmat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "repmat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces a fresh tensor handle; fusion treats repmat as a sink.",
};

fn repmat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("repmat")
        .build()
}

#[runtime_builtin(
    name = "repmat",
    category = "array/shape",
    summary = "Replicate arrays by tiling an input across one or more dimensions.",
    keywords = "repmat,tile,replicate,array,gpu",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::shape::repmat"
)]
fn repmat_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(repmat_error("repmat: replication factors must be specified"));
    }
    let raw_reps = parse_replication_factors(&rest)?;
    match value {
        Value::Tensor(t) => {
            let tiled = repmat_tensor(&t, &raw_reps)?;
            Ok(tensor::tensor_into_value(tiled))
        }
        Value::Num(_) | Value::Int(_) => {
            let tensor = tensor::value_into_tensor_for("repmat", value)
                .map_err(|e| repmat_error(e))?;
            let tiled = repmat_tensor(&tensor, &raw_reps)?;
            Ok(tensor::tensor_into_value(tiled))
        }
        Value::Bool(flag) => {
            let logical = LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
                .map_err(|e| repmat_error(format!("repmat: {e}")))?;
            let tiled = repmat_logical(&logical, &raw_reps)?;
            Ok(Value::LogicalArray(tiled))
        }
        Value::LogicalArray(logical) => {
            let tiled = repmat_logical(&logical, &raw_reps)?;
            Ok(Value::LogicalArray(tiled))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| repmat_error(format!("repmat: {e}")))?;
            let tiled = repmat_complex_tensor(&tensor, &raw_reps)?;
            Ok(complex_tensor_into_value(tiled))
        }
        Value::ComplexTensor(ct) => {
            let tiled = repmat_complex_tensor(&ct, &raw_reps)?;
            Ok(Value::ComplexTensor(tiled))
        }
        Value::String(s) => {
            let array =
                StringArray::new(vec![s], vec![1, 1]).map_err(|e| repmat_error(format!("repmat: {e}")))?;
            let tiled = repmat_string_array(&array, &raw_reps)?;
            Ok(Value::StringArray(tiled))
        }
        Value::StringArray(sa) => {
            let tiled = repmat_string_array(&sa, &raw_reps)?;
            Ok(Value::StringArray(tiled))
        }
        Value::CharArray(ca) => {
            let tiled = repmat_char_array(&ca, &raw_reps)?;
            Ok(Value::CharArray(tiled))
        }
        Value::Cell(ca) => {
            let tiled = repmat_cell_array(&ca, &raw_reps)?;
            Ok(Value::Cell(tiled))
        }
        Value::GpuTensor(handle) => Ok(repmat_gpu_tensor(handle, &raw_reps)?),
        other => Err(repmat_error(format!(
            "repmat: unsupported input type {:?}",
            other
        ))),
    }
}

fn parse_replication_factors(args: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Err(repmat_error("repmat: replication factors must be specified"));
    }
    if args.len() == 1 {
        parse_replication_vector(&args[0])
    } else {
        let mut factors = Vec::with_capacity(args.len());
        for (idx, value) in args.iter().enumerate() {
            let factor = parse_replication_scalar(value)?;
            factors.push(factor);
            if factor == 0 && idx + 1 < args.len() {
                // no-op: just allows subsequent arguments to parse.
            }
        }
        Ok(factors)
    }
}

fn parse_replication_vector(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    let tensor = tensor::value_into_tensor_for("repmat", value.clone())
        .map_err(|e| repmat_error(e))?;
    if tensor.data.is_empty() {
        return Err(repmat_error("repmat: replication vector must contain at least one element"));
    }
    let mut factors = Vec::with_capacity(tensor.data.len());
    for (idx, &raw) in tensor.data.iter().enumerate() {
        factors.push(coerce_rep_factor(raw, idx + 1)?);
    }
    Ok(factors)
}

fn parse_replication_scalar(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                Err(repmat_error("repmat: replication factors must be non-negative integers"))
            } else {
                Ok(raw as usize)
            }
        }
        Value::Num(n) => coerce_rep_factor(*n, 1),
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => {
            let tensor = tensor::value_into_tensor_for("repmat", other.clone())
                .map_err(|e| repmat_error(e))?;
            if tensor.data.len() != 1 {
                return Err(repmat_error("repmat: size arguments must be scalars"));
            }
            coerce_rep_factor(tensor.data[0], 1)
        }
    }
}

fn coerce_rep_factor(value: f64, position: usize) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(repmat_error(format!(
            "repmat: replication factor {position} must be finite"
        )));
    }
    let rounded = value.round();
    let tolerance = (f64::EPSILON * value.abs().max(1.0)).min(1e-9);
    if (rounded - value).abs() > tolerance {
        return Err(repmat_error(format!(
            "repmat: replication factor {position} must be an integer"
        )));
    }
    if rounded < 0.0 {
        return Err(repmat_error("repmat: replication factors must be non-negative integers"));
    }
    if rounded > (usize::MAX as f64) {
        return Err(repmat_error(format!(
            "repmat: replication factor {position} exceeds the maximum supported size"
        )));
    }
    Ok(rounded as usize)
}

fn repmat_tensor(tensor: &Tensor, reps: &[usize]) -> crate::BuiltinResult<Tensor> {
    let (data, shape) = repmat_column_major(&tensor.data, &tensor.shape, reps, "repmat")?;
    Tensor::new(data, shape).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_logical(
    logical: &LogicalArray,
    reps: &[usize],
) -> crate::BuiltinResult<LogicalArray> {
    let (data, shape) = repmat_column_major(&logical.data, &logical.shape, reps, "repmat")?;
    LogicalArray::new(data, shape).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_complex_tensor(
    tensor: &ComplexTensor,
    reps: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    let (data, shape) = repmat_column_major(&tensor.data, &tensor.shape, reps, "repmat")?;
    ComplexTensor::new(data, shape).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_string_array(
    sa: &StringArray,
    reps: &[usize],
) -> crate::BuiltinResult<StringArray> {
    let (data, shape) = repmat_column_major(&sa.data, &sa.shape, reps, "repmat")?;
    StringArray::new(data, shape).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_char_array(ca: &CharArray, reps: &[usize]) -> crate::BuiltinResult<CharArray> {
    let (row_factor, col_factor) = compute_2d_reps(reps)?;
    let (data, rows, cols) =
        repmat_row_major(&ca.data, ca.rows, ca.cols, row_factor, col_factor, "repmat")?;
    CharArray::new(data, rows, cols).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_cell_array(
    cell: &CellArray,
    reps: &[usize],
) -> crate::BuiltinResult<CellArray> {
    let (row_factor, col_factor) = compute_2d_reps(reps)?;
    let (rows, cols) = (
        cell.rows.saturating_mul(row_factor),
        cell.cols.saturating_mul(col_factor),
    );
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| repmat_error("repmat: requested output exceeds maximum size"))?;
    if total == 0 {
        return CellArray::new(Vec::new(), rows, cols).map_err(|e| repmat_error(format!("repmat: {e}")));
    }
    let mut values = Vec::with_capacity(total);
    for _ in 0..row_factor {
        for r in 0..cell.rows {
            for _ in 0..col_factor {
                for c in 0..cell.cols {
                    let idx = r * cell.cols + c;
                    values.push((unsafe { &*cell.data[idx].as_raw() }).clone());
                }
            }
        }
    }
    CellArray::new(values, rows, cols).map_err(|e| repmat_error(format!("repmat: {e}")))
}

fn repmat_gpu_tensor(
    handle: GpuTensorHandle,
    reps: &[usize],
) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(tiled) = provider.repmat(&handle, reps) {
            return Ok(Value::GpuTensor(tiled));
        }
        let gathered = gpu_helpers::gather_tensor(&handle)?;
        let tiled = repmat_tensor(&gathered, reps)?;
        let view = HostTensorView {
            data: &tiled.data,
            shape: &tiled.shape,
        };
        match provider.upload(&view) {
            Ok(new_handle) => Ok(Value::GpuTensor(new_handle)),
            Err(_) => Ok(tensor::tensor_into_value(tiled)),
        }
    } else {
        Err(repmat_error("repmat: no acceleration provider is registered"))
    }
}

fn repmat_column_major<T: Clone>(
    data: &[T],
    shape: &[usize],
    reps: &[usize],
    context: &str,
) -> crate::BuiltinResult<(Vec<T>, Vec<usize>)> {
    let orig_rank = if shape.is_empty() { 1 } else { shape.len() };
    let rank = if reps.len() == 1 {
        orig_rank.max(2)
    } else {
        orig_rank.max(reps.len())
    };

    let mut base_shape = vec![1usize; rank];
    for (idx, &dim) in shape.iter().enumerate() {
        if idx < rank {
            base_shape[idx] = dim;
        }
    }

    let mut factors = vec![1usize; rank];
    if reps.len() == 1 {
        factors.fill(reps[0]);
    } else {
        for (idx, &factor) in reps.iter().enumerate() {
            if idx < rank {
                factors[idx] = factor;
            }
        }
    }

    let mut new_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        new_shape.push(base_shape[i].saturating_mul(factors[i]));
    }

    let orig_total = checked_total(&base_shape, context)?;
    if !(orig_total == data.len() || (orig_total == 0 && data.is_empty())) {
        return Err(repmat_error(format!(
            "{context}: internal shape mismatch (expected {orig_total} elements, found {})",
            data.len()
        )));
    }

    let new_total = checked_total(&new_shape, context)?;
    if new_total == 0 {
        return Ok((Vec::new(), new_shape));
    }

    let strides = column_major_strides(&base_shape);
    let mut out = Vec::with_capacity(new_total);
    for idx in 0..new_total {
        let mut rem = idx;
        let mut src_index = 0usize;
        for dim in 0..rank {
            let dim_size = new_shape[dim];
            let coord = rem % dim_size;
            rem /= dim_size;
            let base = base_shape[dim];
            let orig_coord = if base == 0 { 0 } else { coord % base };
            src_index += orig_coord * strides[dim];
        }
        out.push(data[src_index].clone());
    }
    Ok((out, new_shape))
}

fn repmat_row_major<T: Clone>(
    data: &[T],
    rows: usize,
    cols: usize,
    row_factor: usize,
    col_factor: usize,
    context: &str,
) -> crate::BuiltinResult<(Vec<T>, usize, usize)> {
    if rows.checked_mul(cols).unwrap_or(0) != data.len() && !(rows == 0 || cols == 0) {
        return Err(repmat_error(format!(
            "{context}: internal shape mismatch for row-major array"
        )));
    }
    let new_rows = rows.saturating_mul(row_factor);
    let new_cols = cols.saturating_mul(col_factor);
    let total = new_rows
        .checked_mul(new_cols)
        .ok_or_else(|| repmat_error(format!("{context}: requested output exceeds maximum size")))?;
    if total == 0 {
        return Ok((Vec::new(), new_rows, new_cols));
    }
    let mut out = Vec::with_capacity(total);
    for _ in 0..row_factor {
        for r in 0..rows {
            for _ in 0..col_factor {
                for c in 0..cols {
                    let idx = r * cols + c;
                    out.push(data[idx].clone());
                }
            }
        }
    }
    Ok((out, new_rows, new_cols))
}

fn compute_2d_reps(reps: &[usize]) -> crate::BuiltinResult<(usize, usize)> {
    if reps.is_empty() {
        return Err(repmat_error("repmat: replication factors must be specified"));
    }
    if reps.len() == 1 {
        Ok((reps[0], reps[0]))
    } else {
        if reps.len() > 2 && reps[2..].iter().any(|&f| f > 1) {
            return Err(repmat_error(
                "repmat: RunMat currently supports at most two dimensions for char and cell arrays",
            ));
        }
        Ok((
            reps.first().copied().unwrap_or(1),
            reps.get(1).copied().unwrap_or(1),
        ))
    }
}

fn column_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    strides
}

fn checked_total(shape: &[usize], context: &str) -> crate::BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| repmat_error(format!("{context}: requested output exceeds maximum size")))
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repeats_matrix_with_vector_reps() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = repmat_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(
                Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
            )],
        )
        .expect("repmat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 6]);
                let rows = t.shape[0];
                for col in 0..t.shape[1] {
                    let expected = if col % 2 == 0 {
                        vec![1.0, 3.0, 1.0, 3.0]
                    } else {
                        vec![2.0, 4.0, 2.0, 4.0]
                    };
                    let start = col * rows;
                    let end = start + rows;
                    assert_eq!(&t.data[start..end], expected.as_slice());
                }
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_accepts_column_vector_arguments() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let reps = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let result =
            repmat_builtin(Value::Tensor(tensor), vec![Value::Tensor(reps)]).expect("repmat");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![4, 6]),
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_replication_factor_expands_all_dims() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result =
            repmat_builtin(Value::Tensor(row), vec![Value::Int(IntValue::I32(2))]).expect("repmat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 6]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_high_dim_numeric() {
        let base_data: Vec<f64> = (0..6).map(|v| v as f64).collect();
        let tensor = Tensor::new(base_data.clone(), vec![1, 3, 2]).unwrap();
        let result = repmat_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I32(3)),
            ],
        )
        .expect("repmat");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3, 6]);
                let rows = out.shape[0];
                let cols = out.shape[1];
                let depth = out.shape[2];
                for k in 0..depth {
                    for j in 0..cols {
                        for i in 0..rows {
                            let idx = i + rows * (j + cols * k);
                            let base_col = j % 3;
                            let base_depth = k % 2;
                            let base_idx = base_col + 3 * base_depth;
                            assert_eq!(out.data[idx], base_data[base_idx]);
                        }
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_array_replication_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result =
            repmat_builtin(Value::LogicalArray(logical), vec![Value::from(2.0)]).expect("repmat");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![6, 2]);
                let rows = la.shape[0];
                for col in 0..la.shape[1] {
                    let expected = vec![1u8, 0, 1, 1, 0, 1];
                    let start = col * rows;
                    let end = start + rows;
                    assert_eq!(&la.data[start..end], expected.as_slice());
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_bool_scalar_returns_logical() {
        let result =
            repmat_builtin(Value::Bool(true), vec![Value::Int(IntValue::I32(2))]).expect("repmat");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![2, 2]);
                assert!(la.data.iter().all(|&b| b == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_array_replication_tiles_rows_and_cols() {
        let ca = CharArray::new("hi".chars().collect(), 1, 2).unwrap();
        let result = repmat_builtin(
            Value::CharArray(ca),
            vec![Value::from(2.0), Value::from(3.0)],
        )
        .expect("repmat");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 6);
                let text: String = out.data.iter().collect();
                assert_eq!(text, "hihihihihihi");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_char_rejects_high_dim() {
        let ca = CharArray::new("hi".chars().collect(), 1, 2).unwrap();
        let reps = Tensor::new(vec![2.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let err =
            repmat_builtin(Value::CharArray(ca), vec![Value::Tensor(reps)]).expect_err("repmat");
        assert!(err.to_string().contains("two dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zero_replication_yields_empty_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = repmat_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(2))],
        )
        .expect("repmat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 2]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_errors_on_fractional_replication() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            repmat_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).expect_err("repmat err");
        assert!(err.to_string().contains("integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_errors_on_negative_factor() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = repmat_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(-1))])
            .expect_err("repmat err");
        assert!(err.to_string().contains("non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_errors_on_infinite_factor() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = repmat_builtin(Value::Tensor(tensor), vec![Value::Num(f64::INFINITY)])
            .expect_err("repmat err");
        assert!(err.to_string().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_complex_scalar_tiles() {
        let result = repmat_builtin(
            Value::Complex(1.0, -2.0),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("repmat");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                assert!(ct.data.iter().all(|&(re, im)| re == 1.0 && im == -2.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_complex_tensor_tiles_entries() {
        let base = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0)], vec![1, 2]).unwrap();
        let result = repmat_builtin(
            Value::ComplexTensor(base.clone()),
            vec![Value::Int(IntValue::I32(3)), Value::Int(IntValue::I32(2))],
        )
        .expect("repmat");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![3, 4]);
                let rows = ct.rows;
                let cols = ct.cols;
                let base_rows = base.rows;
                let base_cols = base.cols;
                for col in 0..cols {
                    let orig_col = col % base_cols;
                    for row in 0..rows {
                        let orig_row = row % base_rows;
                        let idx = row + col * rows;
                        let expected_idx = orig_row + orig_col * base_rows;
                        assert_eq!(ct.data[idx], base.data[expected_idx]);
                    }
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_cell_array_tiles_values() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let result = repmat_builtin(
            Value::Cell(cell),
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(2))],
        )
        .expect("repmat");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 4);
                for r in 0..out.rows {
                    for c in 0..out.cols {
                        let value = out.get(r, c).expect("cell element");
                        let expected = if c % 2 == 0 { 1.0 } else { 2.0 };
                        match value {
                            Value::Num(n) => assert_eq!(n, expected),
                            other => panic!("expected numeric cell element, got {other:?}"),
                        }
                    }
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                repmat_builtin(Value::GpuTensor(handle), vec![Value::Int(IntValue::I32(2))])
                    .expect("repmat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repmat_string_scalar() {
        let result =
            repmat_builtin(Value::String("runmat".into()), vec![Value::from(2.0)]).expect("repmat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        "runmat".to_string(),
                        "runmat".to_string(),
                        "runmat".to_string(),
                        "runmat".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn repmat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let cpu_value = repmat_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(3))],
        )
        .expect("repmat cpu");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = repmat_builtin(
            Value::GpuTensor(handle),
            vec![Value::Int(IntValue::I32(2)), Value::Int(IntValue::I32(3))],
        )
        .expect("repmat gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
