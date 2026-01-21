//! MATLAB-compatible `cell2mat` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cell2mat",
        builtin_path = "crate::builtins::cells::core::cell2mat"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cell2mat"
category: "cells/core"
keywords: ["cell2mat", "cell arrays", "matrix conversion", "block concatenation", "gpu fallback"]
summary: "Convert the contents of a MATLAB cell array into a dense numeric, logical, complex, or character matrix."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "cell2mat executes on the host. GPU-resident tensors inside the cell array are gathered before concatenation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::cells::core::cell2mat::tests"
  integration: "builtins::cells::core::cell2mat::tests::cell2mat_gpu_cells_are_gathered"
---

# What does the `cell2mat` function do in MATLAB / RunMat?
`cell2mat(C)` concatenates the numeric, logical, complex, or character arrays stored in the
cell array `C` into a single dense MATLAB array. The cell array must form a rectangular block
structure: all cells in the same row share the same number of rows, all cells in the same column
share the same number of columns, and higher-dimensional extents agree everywhere.

## How does the `cell2mat` function behave in MATLAB / RunMat?
- Works for cell arrays whose elements are **numeric**, **logical**, **complex**, or **character**
  arrays (including scalars and empties). Mixed types are rejected.
- RunMat currently represents cell arrays as 2-D grids. The first dimension tiles rows, the second
  tiles columns, and any higher dimensions inside each element must agree exactly across all cells.
- Empty cells contribute zero extent in their tiling dimension while preserving type information.
- The output array uses column-major layout for numeric, logical, and complex data, and matches
  MATLAB character array semantics for text.
- Calling `cell2mat` on an empty cell array returns the empty double matrix `0×0`.

## `cell2mat` Function GPU Execution Behaviour
`cell2mat` is inherently a host operation because MATLAB cell arrays live on the CPU heap.
If a cell element is a GPU tensor (`gpuArray`) RunMat gathers it to host memory before
concatenating. Providers do **not** need to implement bespoke kernels: the builtin terminates
GPU fusion groups, materialises the inputs on the host, and returns a host-resident array.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do NOT need to call `gpuArray` before `cell2mat`. The builtin always gathers
GPU-resident elements to the host so it can stitch the resulting matrix in CPU memory.
Explicitly wrapping each cell in `gpuArray` is harmless, but there is no residency benefit because
the final result is a standard CPU array.

## Examples of using the `cell2mat` function in MATLAB / RunMat

### Converting a 2-by-2 cell array of scalars into a matrix

```matlab
C = {1, 2; 3, 4};
A = cell2mat(C);
```

Expected output:

```matlab
A =
     1     2
     3     4
```

### Concatenating blocks of different column widths

```matlab
C = {[1 2], [3 4 5]; [6 7], [8 9 10]};
A = cell2mat(C);
```

Expected output:

```matlab
A =
     1     2     3     4     5
     6     7     8     9    10
```

### Converting logical cell contents into a logical matrix

```matlab
C = {true(2,1), false(2,1)};
M = cell2mat(C);
```

Expected output:

```matlab
M =
  2×2 logical array
     1     0
     1     0
```

### Handling complex-valued blocks

```matlab
C = {1+2i, [3+4i 5+6i]};
Z = cell2mat(C);
```

Expected output:

```matlab
Z =
   1.0000 + 2.0000i   3.0000 + 4.0000i   5.0000 + 6.0000i
```

### Producing character matrices from cell arrays of character rows

```matlab
C = {'foo', 'bar'; 'baz', 'qux'};
S = cell2mat(C);
```

Expected output:

```matlab
S =
    'foobar'
    'bazqux'
```

### Tiling higher-dimensional numeric blocks

```matlab
C = {ones(2,2,3), 2*ones(2,1,3)};
X = cell2mat(C);
size(X)
```

Expected output:

```matlab
ans =
     2     3     3
```

### Working with empty cells

```matlab
C = {[], []; [], []};
A = cell2mat(C);
size(A)
```

Expected output:

```matlab
ans =
     0     0
```

### Gathering GPU tensors stored inside cells

```matlab
G = gpuArray(ones(4,1));
C = {G, 2*G};
H = cell2mat(C);   % gathered back to host automatically
classUnderlying(H)
```

Expected output:

```matlab
ans =
    'double'
```

## FAQ

### What element types does `cell2mat` support?
Numeric doubles (scalars or arrays), complex doubles, logical values, and character arrays.
Mixed types are not allowed; every populated cell must have the same fundamental type.

### Can I convert a cell array that contains structs or strings?
No. `cell2mat` requires array-like contents. Use specialised functions such as `string` or
`char` converters for string data, or bespoke logic for structs and tables.

### Do the cell contents need identical shapes?
Cells in the same row must share the same number of rows. Cells in the same column must share
the same number of columns. Any higher dimensions must agree across all cells. Violations
produce a descriptive error that mirrors MATLAB's behaviour.

### What happens with empty cells?
Empty cells contribute zero extent along their tiling dimension. For example, if every element in
a row is empty, the resulting matrix has zero rows for that block. Completely empty cell arrays
produce the `0×0` empty double matrix.

### Does `cell2mat` return GPU arrays when inputs are gpuArray elements?
Not yet. RunMat gathers GPU elements to the host before concatenating. Future releases may
introduce GPU-resident cell storage, at which point providers can supply dedicated kernels.

## See Also
[cell](./cell), [mat2cell](./mat2cell), [num2cell](./num2cell), [cellfun](./cellfun)

## Source & Feedback
- The full source code for the implementation of the `cell2mat` function is available at: [`crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/cells/core/cell2mat.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cell2mat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cell2mat",
    op_kind: GpuOpKind::Custom("cell-flatten"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "cell2mat gathers GPU-resident tensors before concatenating; providers do not supply custom kernels.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cell2mat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cell2mat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |_ctx: &FusionExprContext| {
            Err(FusionError::Message(
                "cell2mat terminates fusion; contents are materialised on the host.",
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion sink; fusion planner stops GPU fusion before calling cell2mat.",
};

const IDENT_INVALID_INPUT: &str = "MATLAB:cell2mat:InvalidInput";
const IDENT_INVALID_CONTENTS: &str = "MATLAB:cell2mat:InvalidContents";
const IDENT_SIZE_LIMIT: &str = "MATLAB:cell2mat:SizeExceeded";

fn cell2mat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("cell2mat")
        .build()
}

fn cell2mat_error_with_identifier(message: impl Into<String>, identifier: &str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("cell2mat")
        .with_identifier(identifier)
        .build()
}

#[runtime_builtin(
    name = "cell2mat",
    category = "cells/core",
    summary = "Convert a cell array of numeric, logical, complex, or character blocks into a dense MATLAB array.",
    keywords = "cell2mat,cell,matrix,concatenation",
    accel = "gather",
    builtin_path = "crate::builtins::cells::core::cell2mat"
)]
async fn cell2mat_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Cell(ca) => cell_array_to_matrix(&ca).await,
        other => Err(cell2mat_error_with_identifier(
            format!("cell2mat: expected a cell array input, got {other:?}"),
            IDENT_INVALID_INPUT,
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementKind {
    Numeric,
    Complex,
    Logical,
    Char,
}

#[derive(Clone)]
struct CellEntry {
    kind: ElementKind,
    shape: Vec<usize>,
    data: EntryData,
}

#[derive(Clone)]
enum EntryData {
    Numeric(Vec<f64>),
    Complex(Vec<(f64, f64)>),
    Logical(Vec<u8>),
    Char(Vec<char>),
}

impl EntryData {
    fn len(&self) -> usize {
        match self {
            EntryData::Numeric(data) => data.len(),
            EntryData::Complex(data) => data.len(),
            EntryData::Logical(data) => data.len(),
            EntryData::Char(data) => data.len(),
        }
    }
}

impl CellEntry {
    fn len(&self) -> usize {
        self.data.len()
    }
}

async fn cell_array_to_matrix(ca: &runmat_builtins::CellArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        // Mirror MATLAB's behaviour: empty cell array -> 0x0 double matrix.
        let tensor = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| cell2mat_error(format!("cell2mat: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }

    let cell_shape = vec![ca.rows, ca.cols];
    let rank = cell_shape.len();

    let mut entries: Vec<CellEntry> = Vec::with_capacity(ca.data.len());
    let mut detected_kind: Option<ElementKind> = None;

    for ptr in &ca.data {
        let gathered = gather_if_needed_async(ptr).await?;
        let entry = parse_cell_entry(gathered)?;
        if let Some(kind) = detected_kind {
            if kind != entry.kind {
                return Err(cell2mat_error_with_identifier(
                    "cell2mat: all cell contents must share the same fundamental type",
                    IDENT_INVALID_CONTENTS,
                ));
            }
        } else {
            detected_kind = Some(entry.kind);
        }
        entries.push(entry);
    }

    let element_kind = detected_kind.unwrap_or(ElementKind::Numeric);
    validate_entry_kinds(&entries, element_kind)?;

    let multi_indices: Vec<Vec<usize>> = (0..entries.len())
        .map(|linear| linear_to_multi_row_major(linear, &cell_shape))
        .collect();

    let mut block_sizes: Vec<Vec<usize>> = cell_shape
        .iter()
        .map(|&extent| vec![0usize; extent])
        .collect();
    let mut extra_shape: Option<Vec<usize>> = None;

    for (entry, indices) in entries.iter().zip(multi_indices.iter()) {
        let tile_dims = extend_shape(&entry.shape, rank);
        for dim in 0..rank {
            let size = tile_dims[dim];
            let slot = block_sizes
                .get_mut(dim)
                .and_then(|v| v.get_mut(indices[dim]))
                .expect("valid cell dimension index");
            if *slot == 0 {
                *slot = size;
            } else if *slot != size {
                return Err(cell2mat_error_with_identifier(
                    "cell2mat: all cells in the same row and column must agree on block sizes",
                    IDENT_INVALID_CONTENTS,
                ));
            }
        }

        let entry_extra = if entry.shape.len() > rank {
            entry.shape[rank..].to_vec()
        } else {
            Vec::new()
        };
        if let Some(existing) = &extra_shape {
            if existing.len() != entry_extra.len()
                || existing.iter().zip(&entry_extra).any(|(a, b)| *a != *b)
            {
                return Err(cell2mat_error_with_identifier(
                    "cell2mat: higher-dimensional extents must match across all cells",
                    IDENT_INVALID_CONTENTS,
                ));
            }
        } else {
            extra_shape = Some(entry_extra);
        }
    }

    let extra_dims = extra_shape.unwrap_or_default();
    let mut result_shape = Vec::with_capacity(rank + extra_dims.len());
    for sizes in &block_sizes {
        let sum = sizes
            .iter()
            .try_fold(0usize, |acc, &v| acc.checked_add(v))
            .ok_or_else(|| {
                cell2mat_error_with_identifier(
                    "cell2mat: resulting matrix is too large to represent on this platform",
                    IDENT_SIZE_LIMIT,
                )
            })?;
        result_shape.push(sum);
    }
    result_shape.extend(extra_dims.iter().copied());

    if result_shape.is_empty() {
        result_shape = vec![0, 0];
    }

    if element_kind == ElementKind::Char && result_shape.len() > 2 {
        return Err(cell2mat_error_with_identifier(
            "cell2mat: character cell contents must form a 2-D character array",
            IDENT_INVALID_CONTENTS,
        ));
    }

    let total_elems = total_len(&result_shape).ok_or_else(|| {
        cell2mat_error_with_identifier(
            "cell2mat: resulting matrix is too large to represent on this platform",
            IDENT_SIZE_LIMIT,
        )
    })?;

    let prefix_offsets: Vec<Vec<usize>> =
        block_sizes.iter().map(|sizes| prefix_sums(sizes)).collect();

    match element_kind {
        ElementKind::Numeric => {
            let mut data = vec![0.0f64; total_elems];
            copy_numeric(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut data,
            )?;
            let tensor = Tensor::new(data, result_shape)
                .map_err(|e| cell2mat_error(format!("cell2mat: {e}")))?;
            Ok(Value::Tensor(tensor))
        }
        ElementKind::Complex => {
            let mut data = vec![(0.0f64, 0.0f64); total_elems];
            copy_complex(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut data,
            )?;
            let tensor = ComplexTensor::new(data, result_shape)
                .map_err(|e| cell2mat_error(format!("cell2mat: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
        ElementKind::Logical => {
            let mut data = vec![0u8; total_elems];
            copy_logical(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut data,
            )?;
            let logical = LogicalArray::new(data, result_shape)
                .map_err(|e| cell2mat_error(format!("cell2mat: {e}")))?;
            Ok(Value::LogicalArray(logical))
        }
        ElementKind::Char => {
            let rows = result_shape.first().copied().unwrap_or(0);
            let cols = result_shape.get(1).copied().unwrap_or(1);
            let char_data = copy_chars(&entries, &multi_indices, rows, cols, &block_sizes)?;
            let array = CharArray::new(char_data, rows, cols)
                .map_err(|e| cell2mat_error(format!("cell2mat: {e}")))?;
            Ok(Value::CharArray(array))
        }
    }
}

fn validate_entry_kinds(entries: &[CellEntry], expected: ElementKind) -> BuiltinResult<()> {
    for entry in entries {
        if entry.len() == 0 {
            continue;
        }
        if entry.kind != expected {
            return Err(cell2mat_error_with_identifier(
                "cell2mat: all non-empty cell contents must share the same fundamental type",
                IDENT_INVALID_CONTENTS,
            ));
        }
    }
    Ok(())
}

fn copy_numeric(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut [f64],
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Numeric(ref data) = entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for (linear, value) in data.iter().enumerate() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output[dest_linear] = *value;
        }
    }
    Ok(())
}

fn copy_complex(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut [(f64, f64)],
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Complex(ref data) = entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for (linear, value) in data.iter().enumerate() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output[dest_linear] = *value;
        }
    }
    Ok(())
}

fn copy_logical(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut [u8],
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Logical(ref data) = entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for (linear, value) in data.iter().enumerate() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output[dest_linear] = *value;
        }
    }
    Ok(())
}

fn copy_chars(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    rows: usize,
    cols: usize,
    block_sizes: &[Vec<usize>],
) -> BuiltinResult<Vec<char>> {
    let mut output = vec!['\0'; rows.saturating_mul(cols)];
    let row_prefix = block_sizes
        .first()
        .map(|sizes| prefix_sums(sizes))
        .unwrap_or_else(|| vec![0]);
    let col_prefix = block_sizes
        .get(1)
        .map(|sizes| prefix_sums(sizes))
        .unwrap_or_else(|| vec![0]);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Char(ref data) = entry.data else {
            continue;
        };
        let shape = extend_shape(&entry.shape, 2);
        let row_offset = row_prefix
            .get(multi.first().copied().unwrap_or(0))
            .copied()
            .unwrap_or(0);
        let col_offset = col_prefix
            .get(multi.get(1).copied().unwrap_or(0))
            .copied()
            .unwrap_or(0);

        for (linear, value) in data.iter().enumerate() {
            let local_idx = linear_to_multi_row_major(linear, &shape);
            let dest_row = row_offset + local_idx.first().copied().unwrap_or(0);
            let dest_col = col_offset + local_idx.get(1).copied().unwrap_or(0);
            let dest_linear = dest_row
                .checked_mul(cols)
                .and_then(|v| v.checked_add(dest_col))
                .ok_or_else(|| {
                    cell2mat_error_with_identifier(
                        "cell2mat: resulting character array exceeds supported size",
                        IDENT_SIZE_LIMIT,
                    )
                })?;
            output[dest_linear] = *value;
        }
    }

    Ok(output)
}

fn compute_base_offsets(
    multi: &[usize],
    prefix_offsets: &[Vec<usize>],
    total_rank: usize,
    rank: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut base = vec![0usize; total_rank];
    for dim in 0..rank.min(prefix_offsets.len()) {
        let idx = multi.get(dim).copied().unwrap_or(0);
        let offset = prefix_offsets[dim].get(idx).copied().ok_or_else(|| {
            cell2mat_error_with_identifier(
                "cell2mat: internal offset calculation failed",
                IDENT_SIZE_LIMIT,
            )
        })?;
        base[dim] = offset;
    }
    Ok(base)
}

fn parse_cell_entry(value: Value) -> BuiltinResult<CellEntry> {
    match value {
        Value::Tensor(t) => Ok(CellEntry {
            kind: ElementKind::Numeric,
            shape: normalize_shape(t.shape.clone()),
            data: EntryData::Numeric(t.data.clone()),
        }),
        Value::Num(n) => Ok(CellEntry {
            kind: ElementKind::Numeric,
            shape: vec![1, 1],
            data: EntryData::Numeric(vec![n]),
        }),
        Value::Int(i) => Ok(CellEntry {
            kind: ElementKind::Numeric,
            shape: vec![1, 1],
            data: EntryData::Numeric(vec![i.to_f64()]),
        }),
        Value::Bool(b) => Ok(CellEntry {
            kind: ElementKind::Logical,
            shape: vec![1, 1],
            data: EntryData::Logical(vec![if b { 1 } else { 0 }]),
        }),
        Value::LogicalArray(la) => Ok(CellEntry {
            kind: ElementKind::Logical,
            shape: normalize_shape(la.shape.clone()),
            data: EntryData::Logical(la.data.clone()),
        }),
        Value::Complex(re, im) => Ok(CellEntry {
            kind: ElementKind::Complex,
            shape: vec![1, 1],
            data: EntryData::Complex(vec![(re, im)]),
        }),
        Value::ComplexTensor(ct) => Ok(CellEntry {
            kind: ElementKind::Complex,
            shape: normalize_shape(ct.shape.clone()),
            data: EntryData::Complex(ct.data.clone()),
        }),
        Value::CharArray(ca) => Ok(CellEntry {
            kind: ElementKind::Char,
            shape: vec![ca.rows, ca.cols],
            data: EntryData::Char(ca.data.clone()),
        }),
        Value::Cell(_) => Err(cell2mat_error_with_identifier(
            "cell2mat: nested cell arrays are not supported",
            IDENT_INVALID_CONTENTS,
        )),
        Value::String(_) | Value::StringArray(_) => Err(cell2mat_error_with_identifier(
            "cell2mat: string inputs are not supported; convert to char arrays first",
            IDENT_INVALID_CONTENTS,
        )),
        Value::GpuTensor(_) => Err(cell2mat_error_with_identifier(
            "cell2mat: unexpected GPU tensor after gather; please report this issue",
            IDENT_INVALID_CONTENTS,
        )),
        other => Err(cell2mat_error_with_identifier(
            format!("cell2mat: unsupported cell element type: {other:?}"),
            IDENT_INVALID_CONTENTS,
        )),
    }
}

fn extend_shape(shape: &[usize], min_len: usize) -> Vec<usize> {
    if shape.len() >= min_len {
        shape.to_vec()
    } else {
        let mut extended = shape.to_vec();
        extended.resize(min_len, 1);
        extended
    }
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape.push(1);
        shape.push(1);
    }
    shape
}

fn prefix_sums(values: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(values.len());
    let mut accum = 0usize;
    for &v in values {
        out.push(accum);
        accum += v;
    }
    out
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut accum = 1usize;
    for &dim in shape {
        strides.push(accum);
        accum = accum.saturating_mul(dim.max(1));
    }
    strides
}

fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn linear_to_multi_column_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            coords.push(0);
        } else {
            coords.push(linear % dim);
            linear /= dim;
        }
    }
    coords
}

fn linear_to_multi_row_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; shape.len()];
    for (idx, &dim) in shape.iter().enumerate().rev() {
        if dim == 0 {
            coords[idx] = 0;
        } else {
            coords[idx] = linear % dim;
            linear /= dim;
        }
    }
    coords
}

fn accumulate_linear(base_offsets: &[usize], local_index: &[usize], strides: &[usize]) -> usize {
    local_index
        .iter()
        .enumerate()
        .map(|(dim, &idx)| (base_offsets[dim] + idx) * strides[dim])
        .sum()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn cell2mat_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::cell2mat_builtin(value))
    }

    fn scalar_cell(values: &[f64], rows: usize, cols: usize) -> Value {
        let cells: Vec<Value> = values.iter().map(|&v| Value::Num(v)).collect();
        crate::make_cell(cells, rows, cols).expect("cell")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn simple_numeric_cell() {
        let cell = scalar_cell(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn block_concatenation() {
        let row1_left = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor"));
        let row1_right =
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).expect("tensor"));
        let row2_left = Value::Tensor(Tensor::new(vec![6.0, 7.0], vec![1, 2]).expect("tensor"));
        let row2_right =
            Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("tensor"));
        let cell = crate::make_cell(vec![row1_left, row1_right, row2_left, row2_right], 2, 2)
            .expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 5]);
                assert_eq!(
                    t.data,
                    vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_cell() {
        let a = Value::Bool(true);
        let b = Value::Bool(false);
        let c = Value::Bool(true);
        let d = Value::Bool(false);
        let cell = crate::make_cell(vec![a, b, c, d], 2, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![2, 2]);
                assert_eq!(la.data, vec![1, 1, 0, 0]);
            }
            other => panic!("expected logical array result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_cell() {
        let values = vec![Value::Complex(1.0, 2.0), Value::Complex(3.0, 4.0)];
        let cell = crate::make_cell(values, 1, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert_eq!(ct.data, vec![(1.0, 2.0), (3.0, 4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_cell() {
        let a = Value::CharArray(CharArray::new("hi".chars().collect(), 1, 2).unwrap());
        let b = Value::CharArray(CharArray::new("BY".chars().collect(), 1, 2).unwrap());
        let cell = crate::make_cell(vec![a, b], 2, 1).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 2);
                assert_eq!(arr.data, vec!['h', 'i', 'B', 'Y']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatched_block_sizes_error() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
        let err = cell2mat_builtin(cell).unwrap_err().to_string();
        assert!(err.contains("block sizes"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn higher_dimensional_tiling() {
        let a = Value::Tensor(Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![2.0; 4], vec![2, 1, 2]).unwrap());
        let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 2]);
                assert_eq!(t.data.len(), 12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_cell_returns_empty_double() {
        let cell = crate::make_cell(Vec::new(), 0, 0).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
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
    fn cell2mat_gpu_cells_are_gathered() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell =
                crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
            let result = cell2mat_builtin(cell).expect("cell2mat");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![2, 2]);
                    assert_eq!(t.data, tensor.data);
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cell2mat_wgpu_cells_are_gathered() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let cell = crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, tensor.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
