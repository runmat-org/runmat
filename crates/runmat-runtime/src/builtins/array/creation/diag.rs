//! MATLAB-compatible `diag` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "diag"
category: "array/creation"
keywords: ["diag", "diagonal", "matrix", "extraction", "gpu"]
summary: "Create diagonal matrices from vectors or extract diagonals from matrices."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Keeps results on the GPU when providers implement custom diag hooks; otherwise gathers to the host, materialises the result once, and uploads it back to the device."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::creation::diag::tests"
  integration: "builtins::array::creation::diag::tests::diag_gpu_roundtrip"
---

# What does the `diag` function do in MATLAB / RunMat?
`diag` either constructs a diagonal matrix from a vector (placing the vector on a specified diagonal)
or extracts a diagonal from a matrix. The behaviour matches MATLAB, including support for offsets,
logical inputs, complex values, and character arrays.

## How does the `diag` function behave in MATLAB / RunMat?
- `diag(v)` with a vector `v` returns a square matrix whose main diagonal is `v`.
- `diag(v, k)` places `v` on the `k`-th diagonal: super-diagonals for `k > 0`, sub-diagonals for
  `k < 0`. The output size grows by `abs(k)`.
- `diag(A)` with a matrix `A` returns a column vector containing the main diagonal of `A`.
- `diag(A, k)` extracts the `k`-th diagonal. When the requested diagonal does not exist, an empty
  column vector is returned.
- Logical inputs stay logical; complex inputs stay complex; character arrays preserve padding with
  spaces off the diagonal.
- Higher-dimensional inputs are accepted when trailing dimensions are singleton—only the leading
  2-D slice participates in the diagonal operation.

## `diag` Function GPU Execution Behaviour
When the input lives on the GPU, RunMat calls the acceleration provider's `diag_from_vector` or
`diag_extract` hook (see the GPU spec). Providers that do not expose these hooks fall back to a
host round-trip: the input is gathered once, the diagonal computation runs on the CPU, and the
result is uploaded back to the device. This keeps follow-up operations in the auto-offload planner
on the GPU without requiring manual `gpuArray` calls.

## Examples of using the `diag` function in MATLAB / RunMat

### Creating a diagonal matrix from a vector

```matlab
v = [4 5 6];
D = diag(v);
```

Expected output:

```matlab
D =
     4     0     0
     0     5     0
     0     0     6
```

### Placing a vector on an upper diagonal

```matlab
v = [1 2 3];
U = diag(v, 1);
```

Expected output:

```matlab
U =
     0     1     0     0
     0     0     2     0
     0     0     0     3
     0     0     0     0
```

### Extracting a subdiagonal as a column vector

```matlab
A = [1 2 3; 4 5 6; 7 8 9];
d = diag(A, -1);
```

Expected output:

```matlab
d =
     4
     8
```

### Building a diagonal matrix from a logical mask

```matlab
mask = logical([1 0 1 0]);
M = diag(mask);
```

Expected output:

```matlab
M =
     1     0     0     0
     0     0     0     0
     0     0     1     0
     0     0     0     0
```

### Keeping diagonal results on the GPU

```matlab
G = gpuArray([2; 4; 8]);
D = diag(G);
firstTwo = gather(D(1:2, 1:2));
```

Expected output:

```matlab
firstTwo =
     2     0
     0     4
```

The matrix `D` stays on the GPU; only the inspected submatrix is gathered.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the auto-offload planner keeps residency on the GPU when expressions make use of GPU
providers. Even when the provider lacks `diag_from_vector` / `diag_extract`, the builtin gathers
once on the host, performs the diagonal operation, and re-uploads the result so later GPU-friendly
ops can continue without intervention.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly
bootstrap GPU residency, you can call `gpuArray` to move data to the GPU. That mirrors MATLAB's
behaviour while still allowing RunMat's planner to decide whether the GPU offers an advantage for
the surrounding computation.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution
toolbox separate from the core language, as their toolbox is a separate commercial product,
MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat
users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### Does `diag` always return a square matrix?
Only when the input is a vector. When the input is a matrix, `diag` returns a column vector whose
length matches the requested diagonal.

### What happens if I request a diagonal outside the matrix bounds?
You receive an empty column vector (size `0 × 0`), matching MATLAB's behaviour.

### Can I use `diag` with logical or character arrays?
Yes. Logical inputs produce logical outputs, and character inputs produce padded character arrays
with spaces away from the diagonal.

### Does `diag` support complex numbers?
Complex inputs are supported. The output keeps the real and imaginary parts intact.

### How do offsets work with vectors?
`diag(v, k)` grows the matrix by `abs(k)` and shifts the diagonal up (`k > 0`) or down (`k < 0`).

### Can I place a diagonal inside a non-square matrix?
No. MATLAB (and RunMat) always produces a square matrix when building from a vector.

### What if the vector is empty?
`diag([])` returns a `0 × 0` matrix. `diag([], k)` returns a square matrix of size `abs(k)` filled
with zeros.

### Do GPU results stay on the device?
Yes—providers with diag hooks operate entirely on the GPU. Providers without hooks perform a single
host gather and upload, so downstream fused expressions still see a GPU handle.

### Is the offset argument required to be an integer?
Yes. Non-integer or non-finite offsets raise an error.

### Does `diag` modify the original input?
No. It always returns a new array, leaving the input unchanged.

## See Also
[eye](./eye), [zeros](./zeros), [ones](./ones), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `diag` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/diag.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/diag.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diag",
    op_kind: GpuOpKind::Custom("diag"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("diag_from_vector"),
        ProviderHook::Custom("diag_extract"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement custom diag hooks; runtimes fall back to a host gather + upload when unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "diag is currently not fused; fusion plans gather to host before invoking the builtin.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("diag", DOC_MD);

#[runtime_builtin(
    name = "diag",
    category = "array/creation",
    summary = "Create diagonal matrices from vectors or extract diagonals from matrices.",
    keywords = "diag,diagonal,matrix,extraction,gpu",
    accel = "shape"
)]
fn diag_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let offset = parse_offset(&rest)?;
    match value {
        Value::Tensor(t) => diag_tensor_value(t, offset),
        Value::ComplexTensor(ct) => diag_complex_value(ct, offset),
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("diag: {e}"))?;
            diag_complex_value(tensor, offset)
        }
        Value::LogicalArray(logical) => diag_logical_value(logical, offset),
        Value::CharArray(chars) => diag_char_value(chars, offset),
        Value::GpuTensor(handle) => diag_gpu_value(handle, offset),
        other => {
            let tensor = tensor::value_into_tensor_for("diag", other)?;
            diag_tensor_value(tensor, offset)
        }
    }
}

fn parse_offset(args: &[Value]) -> Result<isize, String> {
    match args.len() {
        0 => Ok(0),
        1 => offset_from_value(&args[0]),
        _ => Err("diag: too many input arguments".to_string()),
    }
}

fn offset_from_value(value: &Value) -> Result<isize, String> {
    match value {
        Value::Int(i) => offset_from_i64(i.to_i64()),
        Value::Num(n) => offset_from_f64(*n),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err("diag: offset must be a scalar value".to_string());
            }
            offset_from_f64(t.data[0])
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("diag: offset must be real-valued".to_string())
        }
        Value::LogicalArray(l) => {
            if l.data.len() != 1 {
                return Err("diag: offset must be a scalar value".to_string());
            }
            offset_from_i64(if l.data[0] != 0 { 1 } else { 0 })
        }
        Value::Bool(b) => offset_from_i64(if *b { 1 } else { 0 }),
        Value::GpuTensor(_) => Err("diag: offset must be a host scalar value".to_string()),
        Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("diag: offset must be numeric".to_string()),
    }
}

fn offset_from_i64(raw: i64) -> Result<isize, String> {
    if raw < isize::MIN as i64 || raw > isize::MAX as i64 {
        Err("diag: offset magnitude is too large".to_string())
    } else {
        Ok(raw as isize)
    }
}

fn offset_from_f64(raw: f64) -> Result<isize, String> {
    if !raw.is_finite() {
        return Err("diag: offset must be finite".to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("diag: offset must be an integer".to_string());
    }
    if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
        return Err("diag: offset magnitude is too large".to_string());
    }
    offset_from_i64(rounded as i64)
}

const DIAG_SIZE_ERR: &str = "diag: result size exceeds limits";

fn diag_matrix_size(len: usize, offset: isize) -> Result<(usize, usize), String> {
    let shift = offset_abs(offset);
    let size = len
        .checked_add(shift)
        .ok_or_else(|| DIAG_SIZE_ERR.to_string())?;
    let total = size
        .checked_mul(size)
        .ok_or_else(|| DIAG_SIZE_ERR.to_string())?;
    Ok((size, total))
}

fn diag_tensor_value(tensor: Tensor, offset: isize) -> Result<Value, String> {
    let out = diag_tensor_to_tensor(tensor, offset)?;
    Ok(tensor::tensor_into_value(out))
}

fn diag_complex_value(ct: ComplexTensor, offset: isize) -> Result<Value, String> {
    let out = diag_complex_to_tensor(ct, offset)?;
    Ok(complex_tensor_into_value(out))
}

fn diag_logical_value(logical: LogicalArray, offset: isize) -> Result<Value, String> {
    let out = diag_logical_to_array(logical, offset)?;
    Ok(Value::LogicalArray(out))
}

fn diag_char_value(chars: CharArray, offset: isize) -> Result<Value, String> {
    let out = diag_char_to_array(chars, offset)?;
    Ok(Value::CharArray(out))
}

fn gpu_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn try_provider_diag(
    handle: &GpuTensorHandle,
    offset: isize,
) -> Result<Option<GpuTensorHandle>, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    ensure_matrix_shape("diag", &handle.shape)?;
    let (rows, cols) = gpu_rows_cols(&handle.shape);

    if is_vector_like(rows, cols, handle.shape.len()) {
        let len = handle
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| DIAG_SIZE_ERR.to_string())?;
        // Propagate size-related errors to match CPU behaviour.
        let _ = diag_matrix_size(len, offset)?;
        if len == 0 {
            return Ok(None);
        }
        match provider.diag_from_vector(handle, offset) {
            Ok(out) => Ok(Some(out)),
            Err(_) => Ok(None),
        }
    } else {
        let diag_len = diagonal_length(rows, cols, offset);
        if diag_len == 0 {
            return Ok(None);
        }
        match provider.diag_extract(handle, offset) {
            Ok(out) => Ok(Some(out)),
            Err(_) => Ok(None),
        }
    }
}

fn diag_gpu_value(handle: GpuTensorHandle, offset: isize) -> Result<Value, String> {
    if let Some(device_value) = try_provider_diag(&handle, offset)? {
        return Ok(Value::GpuTensor(device_value));
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let result = diag_tensor_to_tensor(tensor, offset)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        if let Ok(uploaded) = provider.upload(&view) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }
    Ok(tensor::tensor_into_value(result))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> Result<(), String> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(format!("{name}: input must be 2-D"))
    } else {
        Ok(())
    }
}

fn is_vector_like(rows: usize, cols: usize, shape_len: usize) -> bool {
    rows == 1 || cols == 1 || shape_len <= 1
}

fn diag_tensor_to_tensor(tensor: Tensor, offset: isize) -> Result<Tensor, String> {
    let Tensor {
        data,
        shape,
        rows,
        cols,
    } = tensor;
    ensure_matrix_shape("diag", &shape)?;
    if is_vector_like(rows, cols, shape.len()) {
        diag_from_vector_real(&data, offset)
    } else {
        diag_from_matrix_real(&data, rows, cols, offset)
    }
}

fn diag_complex_to_tensor(ct: ComplexTensor, offset: isize) -> Result<ComplexTensor, String> {
    let ComplexTensor {
        data,
        shape,
        rows,
        cols,
    } = ct;
    ensure_matrix_shape("diag", &shape)?;
    if is_vector_like(rows, cols, shape.len()) {
        diag_from_vector_complex(&data, offset)
    } else {
        diag_from_matrix_complex(&data, rows, cols, offset)
    }
}

fn diag_logical_to_array(logical: LogicalArray, offset: isize) -> Result<LogicalArray, String> {
    let LogicalArray { data, shape } = logical;
    ensure_matrix_shape("diag", &shape)?;
    let rows = shape.get(0).copied().unwrap_or(0);
    let cols = shape
        .get(1)
        .copied()
        .unwrap_or(if shape.len() <= 1 { 1 } else { 0 });
    if is_vector_like(rows, cols, shape.len()) {
        diag_from_vector_logical(&data, offset)
    } else {
        diag_from_matrix_logical(&data, rows, cols, offset)
    }
}

fn diag_char_to_array(chars: CharArray, offset: isize) -> Result<CharArray, String> {
    let CharArray { data, rows, cols } = chars;
    if rows == 1 || cols == 1 {
        diag_from_vector_char(&data, rows, cols, offset)
    } else {
        diag_from_matrix_char(&data, rows, cols, offset)
    }
}

fn diag_from_vector_real(data: &[f64], offset: isize) -> Result<Tensor, String> {
    let len = data.len();
    let (size, total) = diag_matrix_size(len, offset)?;
    let mut out = vec![0.0; total];
    for (idx, &value) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * size] = value;
    }
    Tensor::new(out, vec![size, size]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_matrix_real(
    data: &[f64],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Result<Tensor, String> {
    let diag_len = diagonal_length(rows, cols, offset);
    if diag_len == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("diag: {e}"));
    }
    let mut out = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        let (row, col) = diagonal_source_index(i, offset);
        let idx = row + col * rows;
        out.push(data[idx]);
    }
    Tensor::new(out, vec![diag_len, 1]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_vector_complex(data: &[(f64, f64)], offset: isize) -> Result<ComplexTensor, String> {
    let len = data.len();
    let (size, total) = diag_matrix_size(len, offset)?;
    let mut out = vec![(0.0, 0.0); total];
    for (idx, &(re, im)) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * size] = (re, im);
    }
    ComplexTensor::new(out, vec![size, size]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_matrix_complex(
    data: &[(f64, f64)],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Result<ComplexTensor, String> {
    let diag_len = diagonal_length(rows, cols, offset);
    if diag_len == 0 {
        return ComplexTensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("diag: {e}"));
    }
    let mut out = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        let (row, col) = diagonal_source_index(i, offset);
        let idx = row + col * rows;
        out.push(data[idx]);
    }
    ComplexTensor::new(out, vec![diag_len, 1]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_vector_logical(data: &[u8], offset: isize) -> Result<LogicalArray, String> {
    let len = data.len();
    let (size, total) = diag_matrix_size(len, offset)?;
    let mut out = vec![0u8; total];
    for (idx, &value) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * size] = if value != 0 { 1 } else { 0 };
    }
    LogicalArray::new(out, vec![size, size]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_matrix_logical(
    data: &[u8],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Result<LogicalArray, String> {
    let diag_len = diagonal_length(rows, cols, offset);
    if diag_len == 0 {
        return LogicalArray::new(Vec::new(), vec![0, 0]).map_err(|e| format!("diag: {e}"));
    }
    let mut out = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        let (row, col) = diagonal_source_index(i, offset);
        let idx = row + col * rows;
        out.push(if data[idx] != 0 { 1 } else { 0 });
    }
    LogicalArray::new(out, vec![diag_len, 1]).map_err(|e| format!("diag: {e}"))
}

fn diag_from_vector_char(
    data: &[char],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Result<CharArray, String> {
    let len = if rows == 1 {
        cols
    } else if cols == 1 {
        rows
    } else {
        data.len()
    };
    let (size, total) = diag_matrix_size(len, offset)?;
    let mut out = vec![' '; total];
    for idx in 0..len {
        let ch = element_from_char_vector(data, rows, cols, idx);
        let (row, col) = diagonal_target_index(idx, offset);
        out[row * size + col] = ch;
    }
    CharArray::new(out, size, size).map_err(|e| format!("diag: {e}"))
}

fn diag_from_matrix_char(
    data: &[char],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Result<CharArray, String> {
    let diag_len = diagonal_length(rows, cols, offset);
    if diag_len == 0 {
        return CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("diag: {e}"));
    }
    let mut out = Vec::with_capacity(diag_len);
    for i in 0..diag_len {
        let (row, col) = diagonal_source_index(i, offset);
        let idx = row * cols + col;
        out.push(data[idx]);
    }
    CharArray::new(out, diag_len, 1).map_err(|e| format!("diag: {e}"))
}

fn element_from_char_vector(data: &[char], rows: usize, cols: usize, index: usize) -> char {
    if rows == 1 {
        data[index]
    } else if cols == 1 {
        data[index * cols]
    } else {
        data[index]
    }
}

fn diagonal_length(rows: usize, cols: usize, offset: isize) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }
    if offset >= 0 {
        let k = offset as usize;
        if k >= cols {
            0
        } else {
            rows.min(cols - k)
        }
    } else {
        let k = (-offset) as usize;
        if k >= rows {
            0
        } else {
            (rows - k).min(cols)
        }
    }
}

fn diagonal_target_index(idx: usize, offset: isize) -> (usize, usize) {
    if offset >= 0 {
        (idx, idx + offset as usize)
    } else {
        let shift = (-offset) as usize;
        (idx + shift, idx)
    }
}

fn diagonal_source_index(idx: usize, offset: isize) -> (usize, usize) {
    if offset >= 0 {
        (idx, idx + offset as usize)
    } else {
        let shift = (-offset) as usize;
        (idx + shift, idx)
    }
}

fn offset_abs(offset: isize) -> usize {
    if offset >= 0 {
        offset as usize
    } else {
        let magnitude = -(offset as i128);
        magnitude as usize
    }
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Tensor};

    #[test]
    fn diag_scalar_returns_scalar() {
        let result = diag_builtin(Value::Num(7.0), Vec::new()).expect("diag");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn diag_complex_scalar_roundtrip() {
        let result = diag_builtin(Value::Complex(1.5, -2.25), Vec::new()).expect("diag");
        assert_eq!(result, Value::Complex(1.5, -2.25));
    }

    #[test]
    fn diag_vector_positive_offset() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("diag");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_eq!(out.rows(), 3);
                assert_eq!(out.cols(), 3);
                assert_eq!(out.data, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_matrix_single_element_returns_scalar() {
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();
        let result = diag_builtin(Value::Tensor(tensor), Vec::new()).expect("diag");
        assert_eq!(result, Value::Num(42.0));
    }

    #[test]
    fn diag_vector_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = diag_builtin(Value::Tensor(tensor), Vec::new()).expect("diag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                for i in 0..3 {
                    for j in 0..3 {
                        let idx = i + j * 3;
                        if i == j {
                            assert_eq!(t.data[idx], (i + 1) as f64);
                        } else {
                            assert_eq!(t.data[idx], 0.0);
                        }
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_empty_vector_returns_empty_matrix() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 1]).unwrap();
        let result = diag_builtin(Value::Tensor(tensor), Vec::new()).expect("diag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_empty_vector_with_offset_expands() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 1]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("diag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_matrix_negative_offset() {
        let tensor = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let offset = Value::Int(IntValue::I32(-1));
        let result = diag_builtin(Value::Tensor(tensor), vec![offset]).expect("diag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![4.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_tensor_requires_two_dimensional_input() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = diag_builtin(Value::Tensor(tensor), Vec::new()).expect_err("diag should fail");
        assert!(err.contains("input must be 2-D"));
    }

    #[test]
    fn diag_offset_out_of_range_returns_empty() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(3))]).expect("diag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_offset_non_integer_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = diag_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).expect_err("diag");
        assert!(err.contains("offset must be an integer"));
    }

    #[test]
    fn diag_offset_nan_errors() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            diag_builtin(Value::Tensor(tensor), vec![Value::Num(f64::NAN)]).expect_err("diag");
        assert!(err.contains("offset must be finite"));
    }

    #[test]
    fn diag_logical_vector_creates_square_matrix() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = diag_builtin(Value::LogicalArray(logical), Vec::new()).expect("diag");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                for i in 0..3 {
                    for j in 0..3 {
                        let idx = i + j * 3;
                        if i == j {
                            assert_eq!(out.data[idx], if i % 2 == 0 { 1 } else { 0 });
                        } else {
                            assert_eq!(out.data[idx], 0);
                        }
                    }
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn diag_logical_matrix_extracts_column() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
        let result = diag_builtin(
            Value::LogicalArray(logical),
            vec![Value::Int(IntValue::I32(-1))],
        )
        .expect("diag");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn diag_char_matrix_extracts_column() {
        let chars = CharArray::new("abcd".chars().collect(), 2, 2).unwrap();
        let result = diag_builtin(Value::CharArray(chars), Vec::new()).expect("diag");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                assert_eq!(out.data, vec!['a', 'd']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn diag_char_vector_yields_square_matrix() {
        let chars = CharArray::new_row("az");
        let result = diag_builtin(Value::CharArray(chars), Vec::new()).expect("diag");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['a', ' ', ' ', 'z']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn diag_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = diag_builtin(Value::GpuTensor(handle), Vec::new()).expect("diag");
            match result {
                Value::GpuTensor(out) => {
                    let host = provider.download(&out).expect("download");
                    assert_eq!(host.shape, vec![3, 3]);
                    assert_eq!(host.data, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
                }
                Value::Tensor(t) => {
                    // Provider may not upload; ensure host fallback is correct.
                    assert_eq!(t.shape, vec![3, 3]);
                }
                other => panic!("unexpected value {other:?}"),
            }
        });
    }

    #[test]
    fn diag_requires_numeric_offset() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = diag_builtin(Value::Tensor(tensor), vec![Value::String("two".into())])
            .expect_err("diag should fail");
        assert!(err.contains("offset must be numeric"));
    }

    #[test]
    fn diag_offset_from_scalar_tensor() {
        let vector = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let offset = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        let result = diag_builtin(Value::Tensor(vector), vec![offset]).expect("diag");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_eq!(out.data, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn diag_wgpu_vector_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let cpu = diag_tensor_to_tensor(tensor.clone(), 1).expect("cpu diag");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = diag_builtin(Value::GpuTensor(handle), vec![Value::Int(IntValue::I32(1))])
            .expect("diag");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn diag_wgpu_extract_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu = diag_tensor_to_tensor(tensor.clone(), -1).expect("cpu diag");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = diag_builtin(
            Value::GpuTensor(handle),
            vec![Value::Int(IntValue::I32(-1))],
        )
        .expect("diag");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        assert_eq!(gathered.shape, cpu.shape);
        for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
