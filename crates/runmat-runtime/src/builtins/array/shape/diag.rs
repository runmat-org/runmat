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
#[runmat_macros::register_doc_text(name = "diag")]
pub const DOC_MD: &str = r#"---
title: "diag"
category: "array/shape"
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
  unit: "builtins::array::shape::diag::tests"
  integration: "builtins::array::shape::diag::tests::diag_gpu_roundtrip"
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
- `diag(v, 'vector')` always returns a column vector copy of `v`, even when `v` is already a vector.
- `diag(v, m)` forces an `m × m` square result, while `diag(v, [m n])` creates an explicit
  rectangular size. You can combine both with offsets (e.g. `diag(v, k, [m n])`) when you need a
  wider diagonal band.
- `diag(___, 'logical')` converts the result to a logical array. `diag(___, 'double')` forces a
  double-precision result when inputs are logical.
- `diag(___, 'like', prototype)` matches the numeric flavour and residency of `prototype`
  (including GPU handles).
- Logical inputs stay logical; complex inputs stay complex; character arrays preserve padding with
  spaces off the diagonal.
- Higher-dimensional inputs are accepted when trailing dimensions are singleton—only the leading
  2-D slice participates in the diagonal operation.

## `diag` Function GPU Execution Behaviour
When the input lives on the GPU, RunMat calls the acceleration provider's `diag_from_vector` or
`diag_extract` hook (see the GPU spec). Providers that do not expose these hooks fall back to a
host round-trip: the input is gathered once, the diagonal computation runs on the CPU, and the
result is uploaded back to the device. Size overrides or the `'vector'` option also trigger the
host path because they are not yet exposed through provider hooks. `'like'` requests are honoured
regardless of the path: GPU prototypes stay on the device, while logical or complex prototypes
adjust the element type accordingly.

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

### Returning a vector without creating a matrix

```matlab
v = [10 20 30];
d = diag(v, 'vector');
```

Expected output:

```matlab
d =
    10
    20
    30
```

### Creating a rectangular diagonal matrix with `sz`

```matlab
v = [1 2];
R = diag(v, [2 4]);
```

Expected output:

```matlab
R =
     1     0     0     0
     0     2     0     0
```

### Matching residency and type with `'like'`

```matlab
G = gpuArray([1 3 5]');
D = diag([1 2 3], 'like', G);
```

The matrix `D` resides on the GPU and mirrors the numeric flavour of `G`. Subsequent GPU-friendly
operations can consume `D` without additional transfers.

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
Only when the input is a vector and you do not request otherwise. Use `'vector'` to keep the result
as a column vector, or pass a size vector (e.g. `diag(v, [m n])`) to create rectangular matrices.

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
host gather and upload, so downstream fused expressions still see a GPU handle. When you request
`'like'` with a GPU prototype, the result is uploaded back to the device even if a host fallback was
required.

### Is the offset argument required to be an integer?
Yes. Non-integer or non-finite offsets raise an error.

### Does `diag` modify the original input?
No. It always returns a new array, leaving the input unchanged.

### How do I match another array's type or residency?
Use the `'like'` syntax: `diag(v, 'like', prototype)`. Logical, complex, and GPU prototypes are
respected even when the computation falls back to the CPU path.

### Is single precision supported?
Not yet. Requesting `'single'` currently raises an error. Use `'like'` with an appropriate prototype
once single-precision support lands.

## See Also
[eye](../creation/eye), [zeros](../creation/zeros), [ones](../creation/ones), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `diag` function is available at: [`crates/runmat-runtime/src/builtins/array/shape/diag.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/diag.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
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

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "diag is currently not fused; fusion plans gather to host before invoking the builtin.",
};

#[derive(Clone, Copy, Debug)]
struct MatrixDims {
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiagClass {
    Double,
    Logical,
}

#[derive(Debug, Clone)]
struct DiagOptions {
    offset: isize,
    size: Option<MatrixDims>,
    force_vector: bool,
    class_spec: Option<DiagClass>,
    like_proto: Option<Value>,
}

impl DiagOptions {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut offset: Option<isize> = None;
        let mut dims: Vec<usize> = Vec::new();
        let mut dims_from_vector = false;
        let mut force_vector = false;
        let mut class_spec: Option<DiagClass> = None;
        let mut like_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "vector" => {
                        force_vector = true;
                        idx += 1;
                        continue;
                    }
                    "like" => {
                        if like_proto.is_some() {
                            return Err("diag: multiple 'like' specifications are not supported"
                                .to_string());
                        }
                        if class_spec.is_some() {
                            return Err(
                                "diag: cannot combine 'like' with class specifiers".to_string()
                            );
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("diag: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("diag: cannot combine 'like' with 'logical'".to_string());
                        }
                        class_spec = Some(DiagClass::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("diag: cannot combine 'like' with 'double'".to_string());
                        }
                        class_spec = Some(DiagClass::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "diag: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("diag: unrecognised option '{other}'"));
                    }
                }
            }

            if !dims_from_vector {
                if let Some(vec_dims) = extract_size_vector(&arg)? {
                    if !dims.is_empty() {
                        return Err(
                            "diag: multiple size specifications are not allowed".to_string()
                        );
                    }
                    dims = vec_dims;
                    dims_from_vector = true;
                    idx += 1;
                    continue;
                }
            }

            if offset.is_none() {
                if let Some(off) = try_offset_from_value(&arg)? {
                    offset = Some(off);
                    idx += 1;
                    continue;
                }
            }

            if !dims_from_vector && dims.len() < 2 {
                if let Some(dim) = try_dimension_from_value(&arg)? {
                    dims.push(dim);
                    idx += 1;
                    continue;
                }
            }

            return Err(format!("diag: unexpected argument {arg:?}"));
        }

        if force_vector && !dims.is_empty() {
            return Err(
                "diag: size arguments are not compatible with the 'vector' option".to_string(),
            );
        }

        let size = if dims.is_empty() {
            None
        } else {
            if dims.len() > 2 {
                return Err(
                    "diag: size specification must contain at most two elements".to_string()
                );
            }
            let rows = dims.first().copied().unwrap_or(0);
            let cols = if dims.len() == 1 { rows } else { dims[1] };
            Some(MatrixDims { rows, cols })
        };

        let offset = offset.unwrap_or(0);

        Ok(Self {
            offset,
            size,
            force_vector,
            class_spec,
            like_proto,
        })
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(ca.data.iter().collect::<String>().to_ascii_lowercase())
        }
        _ => None,
    }
}

fn extract_size_vector(value: &Value) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Tensor(t) => {
            let len = t.data.len();
            let is_vector = t.rows() == 1 || t.cols() == 1 || t.shape.len() == 1;
            if !is_vector || len <= 1 {
                return Ok(None);
            }
            let mut dims = Vec::with_capacity(len);
            for &raw in &t.data {
                dims.push(parse_dimension_scalar(raw)?);
            }
            Ok(Some(dims))
        }
        Value::LogicalArray(_) => Ok(None),
        _ => Ok(None),
    }
}

fn try_dimension_from_value(value: &Value) -> Result<Option<usize>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err("diag: size arguments must be non-negative".to_string());
            }
            Ok(Some(raw as usize))
        }
        Value::Num(n) => parse_dimension_scalar(*n).map(Some),
        Value::Tensor(t) if t.data.len() == 1 => parse_dimension_scalar(t.data[0]).map(Some),
        Value::LogicalArray(l) if l.data.len() == 1 => {
            let dim = if l.data[0] != 0 { 1 } else { 0 };
            Ok(Some(dim))
        }
        Value::Bool(flag) => Ok(Some(if *flag { 1 } else { 0 })),
        _ => Ok(None),
    }
}

fn try_offset_from_value(value: &Value) -> Result<Option<isize>, String> {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => offset_from_value(value).map(Some),
        Value::Tensor(t) if t.data.len() == 1 => offset_from_value(value).map(Some),
        Value::LogicalArray(l) if l.data.len() == 1 => offset_from_value(value).map(Some),
        _ => Ok(None),
    }
}

fn parse_dimension_scalar(raw: f64) -> Result<usize, String> {
    if !raw.is_finite() {
        return Err("diag: size arguments must be finite".to_string());
    }
    if raw < 0.0 {
        return Err("diag: size arguments must be non-negative".to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("diag: size arguments must be integers".to_string());
    }
    Ok(rounded as usize)
}

fn resolve_vector_dims(len: usize, options: &DiagOptions) -> Result<MatrixDims, String> {
    if let Some(spec) = options.size {
        ensure_vector_fits(len, spec, options.offset)?;
        return Ok(spec);
    }
    let shift = offset_abs(options.offset);
    let size = len
        .checked_add(shift)
        .ok_or_else(|| DIAG_SIZE_ERR.to_string())?;
    Ok(MatrixDims {
        rows: size,
        cols: size,
    })
}

fn ensure_vector_fits(len: usize, dims: MatrixDims, offset: isize) -> Result<(), String> {
    if len == 0 {
        return Ok(());
    }
    let diag_len = diagonal_length(dims.rows, dims.cols, offset);
    if diag_len < len {
        return Err("diag: size arguments are too small for the provided vector".to_string());
    }
    Ok(())
}

fn checked_total_len(dims: MatrixDims) -> Result<usize, String> {
    dims.rows
        .checked_mul(dims.cols)
        .ok_or_else(|| DIAG_SIZE_ERR.to_string())
}

#[runtime_builtin(
    name = "diag",
    category = "array/shape",
    summary = "Create diagonal matrices from vectors or extract diagonals from matrices.",
    keywords = "diag,diagonal,matrix,extraction,gpu",
    accel = "shape"
)]
fn diag_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let options = DiagOptions::parse(rest)?;
    match value {
        Value::Tensor(t) => diag_tensor_value(t, &options),
        Value::ComplexTensor(ct) => diag_complex_value(ct, &options),
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("diag: {e}"))?;
            diag_complex_value(tensor, &options)
        }
        Value::LogicalArray(logical) => diag_logical_value(logical, &options),
        Value::CharArray(chars) => diag_char_value(chars, &options),
        Value::GpuTensor(handle) => diag_gpu_value(handle, &options),
        other => {
            let tensor = tensor::value_into_tensor_for("diag", other)?;
            diag_tensor_value(tensor, &options)
        }
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

fn diag_tensor_value(tensor: Tensor, options: &DiagOptions) -> Result<Value, String> {
    let Tensor {
        data,
        shape,
        rows,
        cols,
        ..
    } = tensor;
    ensure_matrix_shape("diag", &shape)?;
    if is_vector_like(rows, cols, shape.len()) {
        if options.force_vector {
            let len = data.len();
            let vector =
                Tensor::new(data.clone(), vec![len, 1]).map_err(|e| format!("diag: {e}"))?;
            return apply_template(tensor::tensor_into_value(vector), options);
        }
        let dims = resolve_vector_dims(data.len(), options)?;
        let out = diag_from_vector_real(&data, options.offset, dims)?;
        apply_template(tensor::tensor_into_value(out), options)
    } else {
        if options.size.is_some() {
            return Err(
                "diag: size arguments are only valid when the input is a vector".to_string(),
            );
        }
        let out = diag_from_matrix_real(&data, rows, cols, options.offset)?;
        apply_template(tensor::tensor_into_value(out), options)
    }
}

fn diag_complex_value(ct: ComplexTensor, options: &DiagOptions) -> Result<Value, String> {
    let ComplexTensor {
        data,
        shape,
        rows,
        cols,
    } = ct;
    ensure_matrix_shape("diag", &shape)?;
    if is_vector_like(rows, cols, shape.len()) {
        if options.force_vector {
            let len = data.len();
            let column =
                ComplexTensor::new(data.clone(), vec![len, 1]).map_err(|e| format!("diag: {e}"))?;
            return apply_template(complex_tensor_into_value(column), options);
        }
        let dims = resolve_vector_dims(data.len(), options)?;
        let out = diag_from_vector_complex(&data, options.offset, dims)?;
        apply_template(complex_tensor_into_value(out), options)
    } else {
        if options.size.is_some() {
            return Err(
                "diag: size arguments are only valid when the input is a vector".to_string(),
            );
        }
        let out = diag_from_matrix_complex(&data, rows, cols, options.offset)?;
        apply_template(complex_tensor_into_value(out), options)
    }
}

fn diag_logical_value(logical: LogicalArray, options: &DiagOptions) -> Result<Value, String> {
    let LogicalArray { data, shape } = logical;
    ensure_matrix_shape("diag", &shape)?;
    let rows = shape.first().copied().unwrap_or(0);
    let cols = shape
        .get(1)
        .copied()
        .unwrap_or(if shape.len() <= 1 { 1 } else { 0 });
    if is_vector_like(rows, cols, shape.len()) {
        if options.force_vector {
            let len = data.len();
            let column =
                LogicalArray::new(data.clone(), vec![len, 1]).map_err(|e| format!("diag: {e}"))?;
            return apply_template(Value::LogicalArray(column), options);
        }
        let dims = resolve_vector_dims(data.len(), options)?;
        let out = diag_from_vector_logical(&data, options.offset, dims)?;
        apply_template(Value::LogicalArray(out), options)
    } else {
        if options.size.is_some() {
            return Err(
                "diag: size arguments are only valid when the input is a vector".to_string(),
            );
        }
        let out = diag_from_matrix_logical(&data, rows, cols, options.offset)?;
        apply_template(Value::LogicalArray(out), options)
    }
}

fn diag_char_value(chars: CharArray, options: &DiagOptions) -> Result<Value, String> {
    if options.class_spec.is_some() || options.like_proto.is_some() {
        return Err("diag: class modifiers are not supported for character inputs".to_string());
    }
    let CharArray { data, rows, cols } = chars;
    if rows == 1 || cols == 1 {
        if options.force_vector {
            let len = if rows == 1 { cols } else { rows };
            let column = CharArray::new(data.clone(), len, 1).map_err(|e| format!("diag: {e}"))?;
            return Ok(Value::CharArray(column));
        }
        let dims = resolve_vector_dims(data.len(), options)?;
        let out = diag_from_vector_char(&data, rows, cols, options.offset, dims)?;
        Ok(Value::CharArray(out))
    } else {
        if options.size.is_some() {
            return Err(
                "diag: size arguments are only valid when the input is a vector".to_string(),
            );
        }
        let out = diag_from_matrix_char(&data, rows, cols, options.offset)?;
        Ok(Value::CharArray(out))
    }
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
    options: &DiagOptions,
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
        if len == 0 || options.force_vector || options.size.is_some() {
            return Ok(None);
        }
        match provider.diag_from_vector(handle, options.offset) {
            Ok(out) => Ok(Some(out)),
            Err(_) => Ok(None),
        }
    } else {
        let diag_len = diagonal_length(rows, cols, options.offset);
        if diag_len == 0 {
            return Ok(None);
        }
        match provider.diag_extract(handle, options.offset) {
            Ok(out) => Ok(Some(out)),
            Err(_) => Ok(None),
        }
    }
}

fn diag_gpu_value(handle: GpuTensorHandle, options: &DiagOptions) -> Result<Value, String> {
    if let Some(device_value) = try_provider_diag(&handle, options)? {
        return apply_template(Value::GpuTensor(device_value), options);
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    diag_tensor_value(tensor, options)
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

fn diag_from_vector_real(data: &[f64], offset: isize, dims: MatrixDims) -> Result<Tensor, String> {
    let total = checked_total_len(dims)?;
    let mut out = vec![0.0; total];
    for (idx, &value) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * dims.rows] = value;
    }
    Tensor::new(out, vec![dims.rows, dims.cols]).map_err(|e| format!("diag: {e}"))
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

fn diag_from_vector_complex(
    data: &[(f64, f64)],
    offset: isize,
    dims: MatrixDims,
) -> Result<ComplexTensor, String> {
    let total = checked_total_len(dims)?;
    let mut out = vec![(0.0, 0.0); total];
    for (idx, &(re, im)) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * dims.rows] = (re, im);
    }
    ComplexTensor::new(out, vec![dims.rows, dims.cols]).map_err(|e| format!("diag: {e}"))
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

fn diag_from_vector_logical(
    data: &[u8],
    offset: isize,
    dims: MatrixDims,
) -> Result<LogicalArray, String> {
    let total = checked_total_len(dims)?;
    let mut out = vec![0u8; total];
    for (idx, &value) in data.iter().enumerate() {
        let (row, col) = diagonal_target_index(idx, offset);
        out[row + col * dims.rows] = if value != 0 { 1 } else { 0 };
    }
    LogicalArray::new(out, vec![dims.rows, dims.cols]).map_err(|e| format!("diag: {e}"))
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
    dims: MatrixDims,
) -> Result<CharArray, String> {
    let len = if rows == 1 {
        cols
    } else if cols == 1 {
        rows
    } else {
        data.len()
    };
    let total = checked_total_len(dims)?;
    let mut out = vec![' '; total];
    for idx in 0..len {
        let ch = element_from_char_vector(data, rows, cols, idx);
        let (row, col) = diagonal_target_index(idx, offset);
        if row < dims.rows && col < dims.cols {
            out[row + col * dims.rows] = ch;
        }
    }
    CharArray::new(out, dims.rows, dims.cols).map_err(|e| format!("diag: {e}"))
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

fn apply_template(value: Value, options: &DiagOptions) -> Result<Value, String> {
    if let Some(proto) = options.like_proto.as_ref() {
        return convert_like(proto, value);
    }
    if let Some(class_spec) = options.class_spec {
        return match class_spec {
            DiagClass::Double => convert_to_double(value),
            DiagClass::Logical => convert_to_logical(value),
        };
    }
    Ok(value)
}

fn convert_like(proto: &Value, value: Value) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => convert_to_logical(value),
        Value::ComplexTensor(_) | Value::Complex(_, _) => convert_to_complex(value),
        Value::GpuTensor(handle) => {
            let host_value = convert_to_double(value)?;
            convert_value_to_gpu(handle, host_value)
        }
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => convert_to_double(value),
        Value::CharArray(_) => Err("diag: cannot use 'like' with character prototypes".to_string()),
        other => {
            let gathered =
                crate::dispatcher::gather_if_needed(other).map_err(|e| format!("diag: {e}"))?;
            convert_like(&gathered, value)
        }
    }
}

fn convert_to_double(value: Value) -> Result<Value, String> {
    match value {
        Value::Tensor(_) | Value::Num(_) => Ok(value),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)?;
            Ok(tensor::tensor_into_value(tensor))
        }
        Value::Bool(flag) => Ok(Value::Num(if flag { 1.0 } else { 0.0 })),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            convert_to_double(tensor::tensor_into_value(tensor))
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            Err("diag: cannot convert complex results to double precision".to_string())
        }
        Value::CharArray(_) => {
            Err("diag: cannot convert character results to double precision".to_string())
        }
        other => Ok(other),
    }
}

fn convert_to_logical(value: Value) -> Result<Value, String> {
    match value {
        Value::LogicalArray(_) | Value::Bool(_) => Ok(value),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Int(i) => Ok(Value::Bool(i.to_i64() != 0)),
        Value::Tensor(tensor) => {
            let mut data = Vec::with_capacity(tensor.data.len());
            for &v in &tensor.data {
                data.push(if v != 0.0 { 1 } else { 0 });
            }
            let logical =
                LogicalArray::new(data, tensor.shape).map_err(|e| format!("diag: {e}"))?;
            Ok(Value::LogicalArray(logical))
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            convert_to_logical(tensor::tensor_into_value(tensor))
        }
        Value::ComplexTensor(tensor) => {
            let mut data = Vec::with_capacity(tensor.data.len());
            for &(re, im) in &tensor.data {
                let is_nonzero = re != 0.0 || im != 0.0;
                data.push(if is_nonzero { 1 } else { 0 });
            }
            let logical =
                LogicalArray::new(data, tensor.shape).map_err(|e| format!("diag: {e}"))?;
            Ok(Value::LogicalArray(logical))
        }
        Value::Complex(re, im) => Ok(Value::Bool(re != 0.0 || im != 0.0)),
        Value::CharArray(_) => Err("diag: cannot convert character results to logical".to_string()),
        other => Ok(other),
    }
}

fn convert_to_complex(value: Value) -> Result<Value, String> {
    match value {
        Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Int(i) => Ok(Value::Complex(i.to_f64(), 0.0)),
        Value::Tensor(tensor) => {
            let data = tensor.data.iter().map(|&re| (re, 0.0)).collect();
            let complex =
                ComplexTensor::new(data, tensor.shape).map_err(|e| format!("diag: {e}"))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::LogicalArray(array) => {
            let data = array
                .data
                .iter()
                .map(|&b| if b != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                .collect();
            let complex =
                ComplexTensor::new(data, array.shape).map_err(|e| format!("diag: {e}"))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::Bool(flag) => Ok(Value::Complex(if flag { 1.0 } else { 0.0 }, 0.0)),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            convert_to_complex(tensor::tensor_into_value(tensor))
        }
        Value::CharArray(_) => Err("diag: cannot convert character results to complex".to_string()),
        other => Ok(other),
    }
}

fn convert_value_to_gpu(_proto: &GpuTensorHandle, value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(_) => Ok(value),
        Value::Tensor(tensor) => upload_tensor_to_gpu(tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("diag: {e}"))?;
            upload_tensor_to_gpu(tensor)
        }
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)?;
            upload_tensor_to_gpu(tensor)
        }
        Value::Bool(flag) => {
            let tensor = Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("diag: {e}"))?;
            upload_tensor_to_gpu(tensor)
        }
        other => Ok(other),
    }
}

fn upload_tensor_to_gpu(tensor: Tensor) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(uploaded) = provider.upload(&view) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }
    Ok(tensor::tensor_into_value(tensor))
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

    #[cfg(feature = "wgpu")]
    fn cpu_diag_tensor(tensor: Tensor, rest: Vec<Value>) -> Tensor {
        let options = super::DiagOptions::parse(rest).expect("options");
        let value = super::diag_tensor_value(tensor, &options).expect("diag");
        crate::builtins::common::tensor::value_into_tensor_for("diag", value)
            .expect("tensor conversion")
    }

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
        assert!(err.contains("unrecognised option"));
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
    fn diag_vector_option_returns_column() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::from("vector")]).expect("diag");
        match result {
            Value::Tensor(column) => {
                assert_eq!(column.shape, vec![3, 1]);
                assert_eq!(column.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor column, got {other:?}"),
        }
    }

    #[test]
    fn diag_size_vector_allocates_rectangular() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let size_vec = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::Tensor(size_vec)]).expect("diag");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 4]);
                assert_eq!(out.data, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_size_too_small_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let size_vec = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let err = diag_builtin(Value::Tensor(tensor), vec![Value::Tensor(size_vec)])
            .expect_err("diag should fail");
        assert!(err.contains("size arguments are too small"));
    }

    #[test]
    fn diag_vector_option_disallows_size_override() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let size_vec = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let err = diag_builtin(
            Value::Tensor(tensor),
            vec![Value::from("vector"), Value::Tensor(size_vec)],
        )
        .expect_err("diag should fail");
        assert!(err.contains("not compatible with the 'vector' option"));
    }

    #[test]
    fn diag_class_logical_yields_logical_output() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let result =
            diag_builtin(Value::Tensor(tensor), vec![Value::from("logical")]).expect("diag");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_eq!(out.data, vec![1, 0, 0, 0, 0, 0, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn diag_class_double_from_logical() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result =
            diag_builtin(Value::LogicalArray(logical), vec![Value::from("double")]).expect("diag");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 3]);
                assert_eq!(out.data, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn diag_single_option_not_supported() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = diag_builtin(Value::Tensor(tensor), vec![Value::from("single")])
            .expect_err("diag should fail");
        assert!(err.contains("single precision"));
    }

    #[test]
    fn diag_like_logical_matches_class() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
        let proto = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let result = diag_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::LogicalArray(proto)],
        )
        .expect("diag");
        assert!(matches!(result, Value::LogicalArray(_)));
    }

    #[test]
    fn diag_like_char_errors() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let proto = CharArray::new_row("a");
        let err = diag_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), Value::CharArray(proto)],
        )
        .expect_err("diag should fail");
        assert!(err.contains("character prototypes"));
    }

    #[test]
    fn diag_like_gpu_returns_gpu_value() {
        test_support::with_test_provider(|provider| {
            let vector = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto_host = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_handle = provider
                .upload(&HostTensorView {
                    data: &proto_host.data,
                    shape: &proto_host.shape,
                })
                .expect("upload prototype");
            let result = diag_builtin(
                Value::Tensor(vector),
                vec![Value::from("like"), Value::GpuTensor(proto_handle)],
            )
            .expect("diag");
            match result {
                Value::GpuTensor(handle) => {
                    let host = provider.download(&handle).expect("download");
                    assert_eq!(host.shape, vec![2, 2]);
                    assert_eq!(host.data, vec![1.0, 0.0, 0.0, 2.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn diag_wgpu_vector_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let cpu = cpu_diag_tensor(tensor.clone(), vec![Value::Int(IntValue::I32(1))]);
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
        let cpu = cpu_diag_tensor(tensor.clone(), vec![Value::Int(IntValue::I32(-1))]);
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
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
