//! MATLAB-compatible `cat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow, RuntimeError};
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cat",
        builtin_path = "crate::builtins::array::shape::cat"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cat"
category: "array/shape"
keywords: ["cat", "concatenate", "array", "dimension", "gpu"]
summary: "Concatenate arrays along a specified dimension while preserving MATLAB semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Runs on the GPU when the provider offers a native cat hook; otherwise RunMat gathers, concatenates on the host, and uploads the result back to the device."
fusion:
  elementwise: false
  reduction: false
  max_inputs: null
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::cat::tests"
  integration: "builtins::array::shape::cat::tests::cat_gpu_roundtrip"
---

# What does the `cat` function do in MATLAB / RunMat?
`cat(dim, A1, A2, …)` concatenates arrays along the dimension `dim`, producing a new
array whose slices along that dimension are the input arrays. The result preserves
column-major ordering and shares MATLAB's rules for implicit singleton expansion.

## How does the `cat` function behave in MATLAB / RunMat?
- `dim` is 1-based and must be a positive integer.
- All inputs must be the same class (double, logical, complex, char, string, or cell).
- Dimensions other than `dim` must match exactly; missing higher dimensions are treated
  as size `1` automatically.
- The size along `dim` becomes the sum of the corresponding sizes from each input.
- Empty inputs participate naturally—if any dimension is zero, the result is empty.
- `gpuArray` inputs stay on the device when an acceleration provider is registered.
- Append an optional `'like', prototype` pair to request output that matches the
  prototype's device residency; numeric prototypes may be host tensors or `gpuArray`
  handles, while logical prototypes must remain on the CPU.

## `cat` Function GPU Execution Behaviour
When every input is a `gpuArray`, RunMat first calls the active provider's
`AccelProvider::cat` hook to concatenate the buffers directly on the device. Providers
that do not expose this hook trigger a transparent fallback: the operands are gathered
to the host, concatenated with the same rules as CPU arrays, and uploaded back to the
originating device so downstream work still sees a `gpuArray`. The `'like'` prototype
is honoured during fallback, ensuring the final residency matches your request. Mixing
host arrays with `gpuArray` inputs is not supported—convert explicitly with
`gpuArray` / `gather` to control residency.

## Examples of using the `cat` function in MATLAB / RunMat

### Concatenating matrices by stacking rows
```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];
C = cat(1, A, B);
```
Expected output:
```matlab
C =
     1     2
     3     4
     5     6
     7     8
```

### Concatenating matrices by appending columns
```matlab
left = [1 3; 2 4];
right = [10 30; 20 40];
wide = cat(2, left, right);
```
Expected output:
```matlab
wide =
     1     3    10    30
     2     4    20    40
```

### Building a 3-D array from 2-D slices
```matlab
slice1 = magic(3);
slice2 = eye(3);
cube = cat(3, slice1, slice2);
```
Expected behaviour: `size(cube)` is `[3 3 2]` with the original matrices in the third dimension.

### Concatenating logical masks without type changes
```matlab
row = logical([1 0 1]);
mask = cat(1, row, ~row);
```
Expected output:
```matlab
mask =
   1   0   1
   0   1   0
```

### Joining character arrays into wider text rows
```matlab
lhs = ['Run' ; 'GPU'];
rhs = ['Mat'; 'Fun'];
words = cat(2, lhs, rhs);
```
Expected output:
```matlab
words =
    RunMat
    GPUFun
```

### Concatenating string arrays along rows
```matlab
names = ["alpha" "beta"];
more = ["gamma" "delta"];
combined = cat(1, names, more);
```
Expected behaviour: `combined` is a 2×2 string array.

### Appending cell array columns for table-like data
```matlab
cols1 = {1, 2; 'a', 'b'};
cols2 = {3, 4; 'c', 'd'};
tableCells = cat(2, cols1, cols2);
```
Expected behaviour: `tableCells` is 2×4 with cells from both inputs interleaved by row.

### Keeping gpuArray inputs on the device
```matlab
G1 = gpuArray(rand(256, 256));
G2 = gpuArray(rand(256, 256));
stacked = cat(3, G1, G2);
```
Expected behaviour: `stacked` remains a `gpuArray` with shape `[256 256 2]`.

### Requesting GPU output with the `'like'` prototype
```matlab
G = gpuArray(rand(3, 3));
H = cat(3, zeros(3, 3), ones(3, 3), "like", G);
```
Expected behaviour: `H` remains a `gpuArray` and `size(H)` returns `[3 3 2]`.

### Concatenating complex arrays preserves imaginary parts
```matlab
z1 = complex([1 2], [3 4]);
z2 = complex([5 6], [7 8]);
joined = cat(2, z1, z2);
```
Expected behaviour: `joined` is complex and retains the real/imaginary components.

### Combining empty inputs yields an empty result
```matlab
emptyRow = zeros(0, 3);
combo = cat(1, emptyRow, emptyRow);
```
Expected behaviour: `combo` is still `0×3`.

## FAQ

**How many input arrays do I need to provide?**  At least two—`cat` requires
`cat(dim, A, B, …)`.

**Can I mix different classes (e.g. doubles and logicals)?**  No. MATLAB requires all
inputs to share the same class. RunMat mirrors this requirement and raises an error
when classes differ.

**Does `cat` support higher dimensions automatically?**  Yes. Missing dimensions are
treated as size `1`, and the output length matches the largest dimension index used.

**What happens with empty inputs?**  Empty arrays participate naturally. If any shared
dimension is zero, the output becomes empty as well.

**How do I ensure the result lives on the GPU without converting every input?**
Append `'like', prototype` to the argument list, where `prototype` is an existing
`gpuArray`. RunMat uploads the concatenated result to the same device when the active
provider lacks a native concatenation kernel.

**Can I concatenate `gpuArray` inputs with regular arrays?**  No. Mixes of `gpuArray`
and host arrays raise an error—convert explicitly with `gpuArray` or `gather` first.

**Does concatenation copy data?**  Yes. Concatenation produces a new array; upstream
values remain unchanged. GPU providers minimise transfers by running in place when
possible.

**How does concatenation interact with complex numbers?**  Complex inputs stay complex.
Real inputs promoted during a complex concatenation receive zero imaginary parts.

**Can I concatenate cell arrays or string arrays?**  Yes. `cat` supports both and
enforces MATLAB's row/column rules for those container types.

**What if the provider lacks a GPU implementation?**  RunMat gathers the operands,
concatenates them on the host, uploads the result back to the device, and logs the
fallback so residency stays consistent.

**Is the result always at least two-dimensional?**  Scalars remain scalars. Otherwise,
trailing singleton dimensions are trimmed unless required to preserve the specified
dimension.

## See Also
- [`reshape`](./reshape)
- [`permute`](./permute)
- [`squeeze`](./squeeze)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)

## Source & Feedback
- Full implementation: `crates/runmat-runtime/src/builtins/array/shape/cat.rs`
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose).
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cat",
    op_kind: GpuOpKind::Custom("cat"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to gather + upload when providers lack a native concatenation kernel.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation is a sink and terminates fusion pipelines.",
};

fn cat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("cat").build()
}

fn cat_err(message: impl Into<String>) -> RuntimeControlFlow {
    cat_error(message).into()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CatCategory {
    Numeric,
    Logical,
    Complex,
    Char,
    String,
    Cell,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LikeDevice {
    Host,
    Gpu,
}

#[derive(Clone, Debug)]
struct LikeSpec {
    device: LikeDevice,
    category_hint: Option<CatCategory>,
}

impl Default for LikeSpec {
    fn default() -> Self {
        Self {
            device: LikeDevice::Host,
            category_hint: None,
        }
    }
}

impl LikeSpec {
    fn from_prototype(proto: Value) -> BuiltinResult<Self> {
        match proto {
            Value::GpuTensor(_) => Ok(Self {
                device: LikeDevice::Gpu,
                category_hint: Some(CatCategory::Numeric),
            }),
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Numeric),
            }),
            Value::LogicalArray(_) | Value::Bool(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Logical),
            }),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Complex),
            }),
            other => Err(cat_err(format!(
                "cat: unsupported prototype for 'like' ({other:?}); provide a numeric or gpuArray prototype"
            ))),
        }
    }

    fn ensure_device(&self, category: CatCategory) -> BuiltinResult<()> {
        if matches!(self.device, LikeDevice::Gpu) && !matches!(category, CatCategory::Numeric) {
            return Err(cat_err(
                "cat: GPU 'like' prototypes are only supported for numeric inputs",
            ));
        }
        Ok(())
    }
}

fn extract_like(mut inputs: Vec<Value>) -> BuiltinResult<(Vec<Value>, LikeSpec)> {
    if inputs.len() >= 2 {
        if let Some(keyword) = keyword_of(&inputs[inputs.len() - 2]) {
            if keyword == "like" {
                let prototype = inputs.last().cloned().unwrap();
                if matches!(
                    prototype,
                    Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_)
                ) {
                    // Treat as data to avoid colliding with textual concatenation cases.
                } else if inputs.len() < 4 {
                    // Removing the pair would leave fewer than two inputs; treat as data.
                } else {
                    let spec = LikeSpec::from_prototype(prototype)?;
                    inputs.pop();
                    inputs.pop();
                    return Ok((inputs, spec));
                }
            }
        }
    }
    Ok((inputs, LikeSpec::default()))
}

#[runtime_builtin(
    name = "cat",
    category = "array/shape",
    summary = "Concatenate arrays along a specified dimension while preserving MATLAB semantics.",
    keywords = "cat,concatenate,array,dimension,gpu",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::shape::cat"
)]
fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(cat_err("cat: at least two input arrays are required"));
    }
    let dim_index = tensor::parse_dimension(&dim, "cat").map_err(cat_err)?;
    let dim_zero = dim_index - 1;

    let (inputs, like) = extract_like(rest)?;
    if inputs.len() < 2 {
        return Err(cat_err("cat: at least two input arrays are required"));
    }

    if inputs.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        if !inputs.iter().all(|v| matches!(v, Value::GpuTensor(_))) {
            return Err(cat_err(
                "cat: cannot mix gpuArray inputs with host arrays; convert them first",
            ));
        }
        return cat_gpu_tensors(dim_zero, inputs, &like);
    }

    let category = determine_category(&inputs, &like)?;
    match category {
        CatCategory::String => cat_string_arrays(dim_zero, inputs),
        CatCategory::Char => cat_char_arrays(dim_zero, inputs),
        CatCategory::Cell => cat_cell_arrays(dim_zero, inputs),
        CatCategory::Logical => cat_logical_arrays(dim_zero, inputs, &like),
        CatCategory::Complex => cat_complex_arrays(dim_zero, inputs, &like),
        CatCategory::Numeric => cat_numeric_tensors(dim_zero, inputs, &like),
    }
}

fn determine_category(inputs: &[Value], like: &LikeSpec) -> BuiltinResult<CatCategory> {
    let mut category = infer_category(inputs)?;
    if let Some(hint) = like.category_hint {
        category = match hint {
            CatCategory::Numeric => {
                if matches!(
                    category,
                    CatCategory::String
                        | CatCategory::Char
                        | CatCategory::Cell
                        | CatCategory::Complex
                ) {
                    return Err(cat_err(
                        "cat: 'like' prototype class does not match the input classes",
                    ));
                }
                CatCategory::Numeric
            }
            CatCategory::Logical => {
                if !matches!(category, CatCategory::Logical) {
                    return Err(cat_err(
                        "cat: 'like' logical prototypes require logical inputs",
                    ));
                }
                CatCategory::Logical
            }
            CatCategory::Complex => {
                if matches!(
                    category,
                    CatCategory::String | CatCategory::Char | CatCategory::Cell
                ) {
                    return Err(cat_err(
                        "cat: 'like' complex prototypes require numeric or complex inputs",
                    ));
                }
                CatCategory::Complex
            }
            CatCategory::Char | CatCategory::String | CatCategory::Cell => {
                return Err(cat_err(
                    "cat: 'like' prototypes for char, string, or cell arrays are not supported",
                ));
            }
        };
    }
    like.ensure_device(category)?;
    Ok(category)
}

fn infer_category(inputs: &[Value]) -> BuiltinResult<CatCategory> {
    let mut has_string = false;
    let mut has_char = false;
    let mut has_cell = false;
    let mut has_complex = false;
    let mut has_numeric = false;
    let mut all_logical = true;

    for value in inputs {
        match value {
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => {
                has_numeric = true;
                all_logical = false;
            }
            Value::LogicalArray(_) | Value::Bool(_) => {
                has_numeric = true;
            }
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                has_complex = true;
                has_numeric = true;
                all_logical = false;
            }
            Value::String(_) | Value::StringArray(_) => {
                has_string = true;
                all_logical = false;
            }
            Value::CharArray(_) => {
                has_char = true;
                all_logical = false;
            }
            Value::Cell(_) => {
                has_cell = true;
                all_logical = false;
            }
            Value::GpuTensor(_) => {
                return Err(cat_err(
                    "cat: gpuArray inputs must be concatenated using the GPU path",
                ));
            }
            other => {
                return Err(cat_err(format!(
                    "cat: unsupported input type for concatenation: {other:?}"
                )));
            }
        }
        if !matches!(value, Value::LogicalArray(_) | Value::Bool(_)) {
            all_logical = false;
        }
    }

    if has_string && (has_char || has_cell || has_complex || (has_numeric && !all_logical)) {
        return Err(cat_err(
            "cat: cannot mix string arrays with other classes",
        ));
    }
    if has_char && (has_cell || has_complex || (has_numeric && !all_logical) || has_string) {
        return Err(cat_err(
            "cat: cannot mix char arrays with other classes",
        ));
    }
    if has_cell && (has_complex || (has_numeric && !all_logical) || has_string || has_char) {
        return Err(cat_err(
            "cat: cannot mix cell arrays with other classes",
        ));
    }
    if has_complex && (has_string || has_char || has_cell) {
        return Err(cat_err(
            "cat: cannot mix complex arrays with textual or cell arrays",
        ));
    }

    if has_string {
        Ok(CatCategory::String)
    } else if has_char {
        Ok(CatCategory::Char)
    } else if has_cell {
        Ok(CatCategory::Cell)
    } else if has_complex {
        Ok(CatCategory::Complex)
    } else if all_logical {
        Ok(CatCategory::Logical)
    } else {
        Ok(CatCategory::Numeric)
    }
}

fn finalize_numeric_output(tensor: Tensor, like: &LikeSpec) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Numeric)?;
    match like.device {
        LikeDevice::Host => Ok(tensor::tensor_into_value(tensor)),
        LikeDevice::Gpu => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                cat_err("cat: GPU output requested via 'like' but no acceleration provider is active")
            })?;
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|err| cat_err(format!("cat: failed to upload concatenated tensor: {err}")))?;
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn cat_numeric_tensors(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = tensor::value_into_tensor_for("cat", value).map_err(cat_err)?;
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[f64]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = Tensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    finalize_numeric_output(tensor, like)
}

fn cat_logical_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_logical(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[u8]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let logical = LogicalArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::LogicalArray(logical))
}

fn cat_complex_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    if values.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return Err(cat_err(
            "cat: complex concatenation requires host arrays; convert gpuArray values first",
        ));
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => {
                ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| cat_err(format!("cat: {e}")))?
            }
            other => {
                let real = tensor::value_into_tensor_for("cat", other).map_err(cat_err)?;
                tensor_to_complex(real)?
            }
        };
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[(f64, f64)]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = ComplexTensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cat_char_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    if dim_zero > 1 {
        return Err(cat_err("cat: char arrays only support dimensions 1 or 2"));
    }
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        if let Value::CharArray(ca) = value {
            arrays.push(ca);
        } else {
            return Err(cat_err("cat: expected char arrays"));
        }
    }
    match dim_zero {
        0 => concat_char_rows(arrays),
        _ => concat_char_cols(arrays),
    }
}

fn concat_char_rows(arrays: Vec<CharArray>) -> BuiltinResult<Value> {
    let cols = arrays.first().map(|a| a.cols).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.cols != cols {
            return Err(cat_err(format!(
                "cat: dimension 2 mismatch between input 1 (size {}) and input {} (size {})",
                cols,
                idx + 1,
                arr.cols
            )));
        }
    }
    let total_rows = arrays.iter().map(|a| a.rows).sum();
    if total_rows == 0 || cols == 0 {
        let data = Vec::new();
        let result = CharArray::new(data, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::CharArray(result));
    }
    let mut data = Vec::with_capacity(total_rows * cols);
    for arr in arrays {
        data.extend_from_slice(&arr.data);
    }
    let result = CharArray::new(data, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::CharArray(result))
}

fn concat_char_cols(arrays: Vec<CharArray>) -> BuiltinResult<Value> {
    let rows = arrays.first().map(|a| a.rows).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.rows != rows {
            return Err(cat_err(format!(
                "cat: dimension 1 mismatch between input 1 (size {}) and input {} (size {})",
                rows,
                idx + 1,
                arr.rows
            )));
        }
    }
    let total_cols = arrays.iter().map(|a| a.cols).sum();
    if total_cols == 0 || rows == 0 {
        let data = Vec::new();
        let result = CharArray::new(data, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::CharArray(result));
    }
    let mut data = Vec::with_capacity(rows * total_cols);
    for row in 0..rows {
        for arr in &arrays {
            for col in 0..arr.cols {
                let idx = row * arr.cols + col;
                data.push(arr.data[idx]);
            }
        }
    }
    let result = CharArray::new(data, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::CharArray(result))
}

fn cat_string_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_string_array(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[String]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let array = StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::StringArray(array))
}

fn cat_cell_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    if dim_zero > 1 {
        return Err(cat_err(
            "cat: cell arrays only support concatenation along dimensions 1 or 2",
        ));
    }
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        if let Value::Cell(cell) = value {
            arrays.push(cell);
        } else {
            return Err(cat_err("cat: expected cell arrays"));
        }
    }
    match dim_zero {
        0 => concat_cell_rows(arrays),
        _ => concat_cell_cols(arrays),
    }
}

fn concat_cell_rows(arrays: Vec<CellArray>) -> BuiltinResult<Value> {
    let cols = arrays.first().map(|a| a.cols).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.cols != cols {
            return Err(cat_err(format!(
                "cat: dimension 2 mismatch between input 1 (size {}) and input {} (size {})",
                cols,
                idx + 1,
                arr.cols
            )));
        }
    }
    let total_rows = arrays.iter().map(|a| a.rows).sum();
    if total_rows == 0 || cols == 0 {
        let cell = CellArray::new(Vec::new(), total_rows, cols)
            .map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::Cell(cell));
    }
    let mut values = Vec::with_capacity(total_rows * cols);
    for arr in arrays {
        for handle in arr.data {
            let value = unsafe { &*handle.as_raw() }.clone();
            values.push(value);
        }
    }
    let cell = CellArray::new(values, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::Cell(cell))
}

fn concat_cell_cols(arrays: Vec<CellArray>) -> BuiltinResult<Value> {
    let rows = arrays.first().map(|a| a.rows).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.rows != rows {
            return Err(cat_err(format!(
                "cat: dimension 1 mismatch between input 1 (size {}) and input {} (size {})",
                rows,
                idx + 1,
                arr.rows
            )));
        }
    }
    let total_cols = arrays.iter().map(|a| a.cols).sum();
    if total_cols == 0 || rows == 0 {
        let cell = CellArray::new(Vec::new(), rows, total_cols)
            .map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::Cell(cell));
    }
    let mut values = Vec::with_capacity(rows * total_cols);
    for row in 0..rows {
        for arr in &arrays {
            for col in 0..arr.cols {
                let idx = row * arr.cols + col;
                let value = unsafe { &*arr.data[idx].as_raw() }.clone();
                values.push(value);
            }
        }
    }
    let cell = CellArray::new(values, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::Cell(cell))
}

fn cat_gpu_tensors(dim_zero: usize, values: Vec<Value>, like: &LikeSpec) -> BuiltinResult<Value> {
    if let Some(hint) = like.category_hint {
        if !matches!(hint, CatCategory::Numeric) {
            return Err(cat_err(
                "cat: 'like' prototype class does not match gpuArray inputs",
            ));
        }
    }
    like.ensure_device(CatCategory::Numeric)?;

    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| cat_err("cat: no acceleration provider is registered"))?;

    let mut handles = Vec::with_capacity(values.len());
    for value in values {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle);
        }
    }

    // Native provider hook
    if let Ok(result) = provider.cat(dim_zero + 1, &handles) {
        return finalize_gpu_value(result, like);
    }

    let mut tensors = Vec::with_capacity(handles.len());
    for handle in &handles {
        let tensor = gpu_helpers::gather_tensor(handle)?;
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[f64]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = Tensor::new(data, shape.clone()).map_err(|e| cat_err(format!("cat: {e}")))?;
    if matches!(like.device, LikeDevice::Host) {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let view = HostTensorView {
        data: &tensor.data,
        shape: &shape,
    };
    match provider.upload(&view) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

fn concat_column_major<T: Clone>(
    dim_zero: usize,
    shapes: &[Vec<usize>],
    data: &[&[T]],
    context: &str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    if shapes.is_empty() {
        return Err(cat_err(format!("{context}: no inputs to concatenate")));
    }
    let rank = shapes
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(dim_zero + 1);

    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.clone();
        while current.len() < rank {
            current.push(1);
        }
        padded.push(current);
    }

    for (idx, (shape, slice)) in padded.iter().zip(data.iter()).enumerate() {
        let expected = checked_product(shape)
            .ok_or_else(|| cat_err(format!("{context}: input {} exceeds maximum size", idx + 1)))?;
        if expected != slice.len() {
            return Err(cat_err(format!(
                "{context}: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                slice.len(),
                expected
            )));
        }
    }

    for axis in 0..rank {
        if axis == dim_zero {
            continue;
        }
        let reference = padded[0][axis];
        for (idx, shape) in padded.iter().enumerate().skip(1) {
            if shape[axis] != reference {
                return Err(cat_err(format!(
                    "{context}: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                )));
            }
        }
    }

    let mut output_shape = padded[0].clone();
    let mut concat_dim = 0usize;
    for shape in &padded {
        concat_dim = concat_dim
            .checked_add(shape[dim_zero])
            .ok_or_else(|| cat_err(format!("{context}: concatenated dimension exceeds maximum size")))?;
    }
    output_shape[dim_zero] = concat_dim;

    let total = match checked_product(&output_shape) {
        Some(total) => total,
        None => return Err(cat_err(format!("{context}: resulting array exceeds maximum size"))),
    };
    if total == 0 {
        return Ok((Vec::new(), normalize_shape(output_shape, dim_zero)));
    }

    let inner = if dim_zero == 0 {
        1
    } else {
        output_shape[..dim_zero].iter().product()
    };
    let outer = if dim_zero + 1 >= rank {
        1
    } else {
        output_shape[dim_zero + 1..].iter().product()
    };

    let mut output = Vec::with_capacity(total);
    for outer_idx in 0..outer {
        for (shape, slice) in padded.iter().zip(data.iter()) {
            let mid = shape[dim_zero];
            let chunk = mid * inner;
            if chunk == 0 {
                continue;
            }
            let offset = outer_idx * chunk;
            output.extend_from_slice(&slice[offset..offset + chunk]);
        }
    }

    Ok((output, normalize_shape(output_shape, dim_zero)))
}

fn normalize_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    let min_len = (dim_zero + 1).max(2).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn value_into_logical(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!("cat: expected logical inputs, got {:?}", other))),
    }
}

fn value_into_string_array(value: Value) -> BuiltinResult<StringArray> {
    match value {
        Value::StringArray(array) => Ok(array),
        Value::String(text) => {
            StringArray::new(vec![text], vec![1, 1]).map_err(|e| cat_err(format!("cat: {e}")))
        }
        other => Err(cat_err(format!("cat: expected string arrays, got {:?}", other))),
    }
}

fn tensor_to_complex(tensor: Tensor) -> BuiltinResult<ComplexTensor> {
    let data = tensor.data.into_iter().map(|re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape).map_err(|e| cat_err(format!("cat: {e}")))
}

fn finalize_gpu_value(
    handle: runmat_accelerate_api::GpuTensorHandle,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if matches!(like.device, LikeDevice::Host) {
        let tensor = gpu_helpers::gather_tensor(&handle)?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_numeric_rows() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .expect("cat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.contains("dimension 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_char_columns() {
        let left = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let right = CharArray::new("Mat".chars().collect(), 1, 3).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(left), Value::CharArray(right)],
        )
        .expect("cat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text: String = arr.data.iter().collect();
                assert_eq!(text, "RunMat");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_logical_preserves_type() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::LogicalArray(top), Value::LogicalArray(bottom)],
        )
        .expect("cat");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 3]);
                assert_eq!(arr.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload a");
            let hb = provider.upload(&view_b).expect("upload b");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
            )
            .expect("cat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_inputs() {
        test_support::with_test_provider(|provider| {
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![
                    Value::Tensor(a),
                    Value::Tensor(b),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with like");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_logical_mismatch_errors() {
        let proto = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
                Value::from("like"),
                Value::LogicalArray(proto),
            ],
        )
        .unwrap_err();
        assert!(err.contains("logical"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .expect("cat cpu");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
        )
        .expect("cat gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
