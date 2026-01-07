//! MATLAB-compatible `mat2cell` builtin for RunMat.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(all(test, feature = "wgpu"))]
use crate::accel_provider;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{gather_if_needed, make_cell_with_shape};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "mat2cell",
        builtin_path = "crate::builtins::cells::core::mat2cell"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "mat2cell"
category: "cells/core"
keywords: ["mat2cell", "cell array", "partition", "submatrix", "block slicing", "gpu fallback"]
summary: "Split arrays into cell-array blocks using MATLAB-compatible dimension partitions."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "The builtin gathers gpuArray inputs back to the host because providers do not yet expose block-splitting hooks."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::cells::core::mat2cell::tests"
  integration: "builtins::cells::core::mat2cell::tests::mat2cell_gpu_falls_back_to_host"
---

# What does the `mat2cell` function do in MATLAB / RunMat?
`mat2cell` partitions an array along each dimension according to vectors of block sizes and returns
those blocks inside a cell array. The result preserves the element type of the input, and each cell
contains a contiguous slice of the original data.

## How does the `mat2cell` function behave in MATLAB / RunMat?
- Supply one size vector per dimension you want to split. Their elements must be non-negative
  integers that sum to the corresponding dimension of the input array.
- If you omit trailing size vectors, RunMat assumes the remaining dimensions stay intact
  (`size(A, dim)`).
- Zero-sized blocks are allowed and produce empty matrices (or empty arrays of the input type).
- N-dimensional inputs are supported; the output cell array has one dimension per supplied size
  vector.
- Inputs can be numeric, complex, logical, string, or character arrays. Struct, object, and cell
  arrays are not yet supported.

## `mat2cell` Function GPU Execution Behaviour
When the input is a `gpuArray`, RunMat gathers it back to the host before performing the partition,
and the resulting cells contain host tensors. This matches MATLAB semantics except for the
residency—providers do not yet offer on-device block-splitting hooks. Once such hooks become
available, RunMat can keep results on the GPU with no user code changes.

## Examples of using the `mat2cell` function in MATLAB / RunMat

### Splitting a matrix into four quadrants

```matlab
A = reshape(1:16, 4, 4);
C = mat2cell(A, [2 2], [1 3]);
```

Expected output:

```matlab
size(C)        % => [2 2]
double(C{2,2}) % => [7 11 15; 8 12 16]
```

### Splitting a column vector with only row sizes

```matlab
v = (1:6)';
blocks = mat2cell(v, [2 1 3]);
```

Expected output:

```matlab
cellfun(@numel, blocks)   % => [2; 1; 3]
blocks{3}                 % => [4; 5; 6]
```

### Partitioning a 3-D tensor

```matlab
T = reshape(1:24, [3 4 2]);
C = mat2cell(T, [1 2], [2 2], [1 1]);
```

Expected output:

```matlab
size(C)                  % => [2 2 2]
double(C{2,1,2}(:,:,1))  % => [14 17; 15 18]
```

### Using zero-sized blocks

```matlab
E = zeros(3, 2);
C = mat2cell(E, [0 3], [1 1]);
cellfun(@size, C, 'UniformOutput', false)
```

Expected output:

```matlab
ans{1,1} = [0 1]
ans{1,2} = [0 1]
ans{2,1} = [3 1]
ans{2,2} = [3 1]
```

### Splitting a character matrix into rows

```matlab
names = ['foo '; 'bar '; 'baz '];
C = mat2cell(names, [1 2], size(names, 2));
```

Expected output:

```matlab
C{1,1}   % => 'foo '
C{2,1}   % => ['bar '; 'baz ']
```

### Working with logical arrays

```matlab
mask = logical([1 0 1; 0 1 0]);
cells = mat2cell(mask, 2, [1 1 1]);
```

Expected output:

```matlab
cells{1,2}   % => logical column vector [0; 1]
class(cells{1,2})  % => 'logical'
```

## GPU residency in RunMat (Do I need `gpuArray`?)
The current implementation gathers GPU inputs to the host, produces host cell arrays, and returns
CPU-resident tensors inside each cell. Explicit `gpuArray` calls are not required; once GPU providers
offer block-splitting hooks, mat2cell will keep results on the device automatically.

## FAQ

### Do the partition vectors have to sum exactly to the dimension size?
Yes. Each size vector must consist of non-negative integers whose sum matches the corresponding
dimension of the input array. RunMat raises an error when the sums differ.

### What happens if I omit trailing dimension vectors?
RunMat mirrors MATLAB: omitted trailing vectors are treated as a single block that covers the entire
dimension (`size(A, dim)`), so many common 2-D use cases only need two vectors.

### Are zero-sized blocks allowed?
Yes. A zero entry in a partition vector produces an empty array in the corresponding cell. This is
useful when you need placeholders that preserve grid structure.

### What element types are supported?
Numeric, complex, logical, string, and character arrays are supported today. Struct arrays, object
arrays, and cell arrays will gain support in a future update.

### Does `mat2cell` copy the data?
Yes. Each cell receives its own copy of the underlying block so that you can modify the cell contents
without affecting other cells or the original array.

## See Also
[cell](./cell), [cell2mat](./cell2mat), [num2cell](./num2cell), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `mat2cell` function is available at: [`crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/cells/core/mat2cell.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::mat2cell")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mat2cell",
    op_kind: GpuOpKind::Custom("container"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "mat2cell gathers gpuArray inputs to the host until providers expose block-splitting hooks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::mat2cell")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mat2cell",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Partitioning into cells terminates fusion; blocks are produced on the host.",
};

#[runtime_builtin(
    name = "mat2cell",
    category = "cells/core",
    summary = "Split arrays into cell-array blocks.",
    keywords = "mat2cell,cell array,partition,block",
    builtin_path = "crate::builtins::cells::core::mat2cell"
)]
fn mat2cell_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Err("mat2cell: expected at least one size vector".to_string());
    }

    let host_value = gather_if_needed(&value).map_err(|e| format!("mat2cell: {e}"))?;
    let mut size_args = Vec::with_capacity(rest.len());
    for arg in rest {
        let gathered = gather_if_needed(&arg).map_err(|e| format!("mat2cell: {e}"))?;
        size_args.push(gathered);
    }

    let input = Mat2CellInput::try_new(host_value)?;
    let partitions = parse_partitions(input.normalized_dims(), &size_args)?;
    split_into_cells(&input, partitions)
}

#[derive(Debug)]
enum Mat2CellKind {
    Tensor(Tensor),
    Complex(ComplexTensor),
    Logical(LogicalArray),
    Strings(StringArray),
    Char(CharArray),
}

struct Mat2CellInput {
    kind: Mat2CellKind,
    base_shape: Vec<usize>,
    normalized_dims: Vec<usize>,
}

impl Mat2CellInput {
    fn try_new(value: Value) -> Result<Self, String> {
        match value {
            Value::Tensor(t) => {
                let base_shape = adapt_numeric_shape(&t.shape);
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Tensor(t),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::ComplexTensor(t) => {
                let base_shape = adapt_numeric_shape(&t.shape);
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Complex(t),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::LogicalArray(l) => {
                let base_shape = adapt_numeric_shape(&l.shape);
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Logical(l),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::String(s) => {
                let array =
                    StringArray::new(vec![s], vec![1, 1]).map_err(|e| format!("mat2cell: {e}"))?;
                let base_shape = vec![1, 1];
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Strings(array),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::StringArray(sa) => {
                let base_shape = if sa.shape.is_empty() {
                    vec![1, sa.rows()]
                } else {
                    sa.shape.clone()
                };
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Strings(sa),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::CharArray(ca) => {
                let base_shape = vec![ca.rows, ca.cols];
                let normalized_dims = normalize_dims(&base_shape);
                Ok(Self {
                    kind: Mat2CellKind::Char(ca),
                    base_shape,
                    normalized_dims,
                })
            }
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| format!("mat2cell: {e}"))?;
                Self::try_new(Value::ComplexTensor(tensor))
            }
            Value::Num(n) => {
                let tensor =
                    tensor::value_into_tensor_for("mat2cell", Value::Num(n))?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            Value::Int(i) => {
                let tensor =
                    tensor::value_into_tensor_for("mat2cell", Value::Int(i.clone()))?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            Value::Bool(b) => {
                let tensor =
                    tensor::value_into_tensor_for("mat2cell", Value::Bool(b))?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            other => Err(format!(
                "mat2cell: unsupported input type {other:?}; expected numeric, logical, string, or char arrays"
            )),
        }
    }

    fn normalized_dims(&self) -> &[usize] {
        &self.normalized_dims
    }

    fn extract(&self, start: &[usize], sizes: &[usize]) -> Result<Value, String> {
        match &self.kind {
            Mat2CellKind::Tensor(t) => {
                let data = copy_block(&t.data, &self.base_shape, start, sizes)?;
                let shape = adjust_output_shape(sizes);
                let tensor = Tensor::new(data, shape).map_err(|e| format!("mat2cell: {e}"))?;
                Ok(tensor::tensor_into_value(tensor))
            }
            Mat2CellKind::Complex(t) => {
                let data = copy_block(&t.data, &self.base_shape, start, sizes)?;
                let shape = adjust_output_shape(sizes);
                if data.len() == 1 {
                    let (re, im) = data[0];
                    Ok(Value::Complex(re, im))
                } else {
                    let tensor =
                        ComplexTensor::new(data, shape).map_err(|e| format!("mat2cell: {e}"))?;
                    Ok(Value::ComplexTensor(tensor))
                }
            }
            Mat2CellKind::Logical(arr) => {
                let data = copy_block(&arr.data, &self.base_shape, start, sizes)?;
                if data.len() == 1 {
                    Ok(Value::Bool(data[0] != 0))
                } else {
                    let shape = adjust_output_shape(sizes);
                    let logical =
                        LogicalArray::new(data, shape).map_err(|e| format!("mat2cell: {e}"))?;
                    Ok(Value::LogicalArray(logical))
                }
            }
            Mat2CellKind::Strings(arr) => {
                let data = copy_block(&arr.data, &self.base_shape, start, sizes)?;
                if data.len() == 1 {
                    Ok(Value::String(data.into_iter().next().unwrap()))
                } else {
                    let shape = adjust_output_shape(sizes);
                    let strings =
                        StringArray::new(data, shape).map_err(|e| format!("mat2cell: {e}"))?;
                    Ok(Value::StringArray(strings))
                }
            }
            Mat2CellKind::Char(ca) => slice_char_array(ca, start, sizes),
        }
    }
}

fn parse_partitions(dims: &[usize], size_args: &[Value]) -> Result<Vec<Vec<usize>>, String> {
    let mut dim_sizes = dims.to_vec();
    let target = dim_sizes.len().max(size_args.len());
    if dim_sizes.len() < target {
        dim_sizes.resize(target, 1);
    }
    let mut partitions = Vec::with_capacity(target);
    for (idx, &dim_size) in dim_sizes.iter().enumerate() {
        if idx < size_args.len() {
            let vec = parse_partition_vector(&size_args[idx], dim_size, idx + 1)?;
            partitions.push(vec);
        } else {
            partitions.push(vec![dim_size]);
        }
    }
    Ok(partitions)
}

fn split_into_cells(input: &Mat2CellInput, partitions: Vec<Vec<usize>>) -> Result<Value, String> {
    let mut per_dim_counts: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
    if per_dim_counts.is_empty() {
        per_dim_counts = vec![1, 1];
    }
    let normalized_shape = normalize_cell_shape(per_dim_counts.clone());
    let total_cells = normalized_shape.iter().product::<usize>();

    if total_cells == 0 || partitions.iter().any(|p| p.is_empty()) {
        return make_cell_with_shape(Vec::new(), normalized_shape)
            .map_err(|e| format!("mat2cell: {e}"));
    }

    let offsets: Vec<Vec<usize>> = partitions.iter().map(|part| prefix_sums(part)).collect();

    let rank = partitions.len();
    let mut indices = vec![0usize; rank];
    let mut cells = Vec::with_capacity(total_cells);

    loop {
        let mut start = Vec::with_capacity(rank);
        let mut sizes = Vec::with_capacity(rank);
        for dim in 0..rank {
            let idx = indices[dim];
            start.push(offsets[dim][idx]);
            sizes.push(partitions[dim][idx]);
        }
        let value = input.extract(&start, &sizes)?;
        cells.push(value);

        let mut carry = true;
        for dim in (0..rank).rev() {
            indices[dim] += 1;
            if indices[dim] < partitions[dim].len() {
                carry = false;
                break;
            }
            indices[dim] = 0;
        }
        if carry {
            break;
        }
    }

    make_cell_with_shape(cells, normalized_shape).map_err(|e| format!("mat2cell: {e}"))
}

fn parse_partition_vector(
    value: &Value,
    dim_size: usize,
    dim_index: usize,
) -> Result<Vec<usize>, String> {
    let numbers = extract_numeric_vector(value).ok_or_else(|| {
        format!(
            "mat2cell: size arguments must be numeric for dimension {}",
            dim_index
        )
    })?;

    if numbers.is_empty() {
        if dim_size == 0 {
            return Ok(Vec::new());
        }
        return Err(format!(
            "mat2cell: partition sizes for dimension {} must sum to {}",
            dim_index, dim_size
        ));
    }

    let mut total: usize = 0;
    let mut parts = Vec::with_capacity(numbers.len());
    for (idx, n) in numbers.iter().enumerate() {
        if !n.is_finite() {
            return Err(format!(
                "mat2cell: size entries must be finite (dimension {}, index {})",
                dim_index,
                idx + 1
            ));
        }
        let rounded = n.round();
        if (rounded - n).abs() > f64::EPSILON {
            return Err(format!(
                "mat2cell: size entries must be integers (dimension {}, index {})",
                dim_index,
                idx + 1
            ));
        }
        if rounded < 0.0 {
            return Err(format!(
                "mat2cell: size entries must be non-negative (dimension {}, index {})",
                dim_index,
                idx + 1
            ));
        }
        let value = rounded as usize;
        total = total
            .checked_add(value)
            .ok_or_else(|| "mat2cell: partition sum exceeds platform limits".to_string())?;
        parts.push(value);
    }

    if total != dim_size {
        return Err(format!(
            "mat2cell: partition sizes for dimension {} must sum to {} (got {})",
            dim_index, dim_size, total
        ));
    }
    Ok(parts)
}

fn extract_numeric_vector(value: &Value) -> Option<Vec<f64>> {
    match value {
        Value::Num(n) => Some(vec![*n]),
        Value::Int(i) => Some(vec![i.to_f64()]),
        Value::Bool(b) => Some(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(t) => {
            if is_vector_shape(&t.shape) {
                Some(t.data.clone())
            } else {
                None
            }
        }
        Value::LogicalArray(arr) => {
            if is_vector_shape(&arr.shape) {
                Some(
                    arr.data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
        if non_singleton > 1 {
            return false;
        }
    }
    true
}

fn copy_block<T: Clone>(
    data: &[T],
    shape: &[usize],
    start: &[usize],
    sizes: &[usize],
) -> Result<Vec<T>, String> {
    let rank = sizes.len();
    let extended_shape = extend_shape(shape, rank);
    let strides = column_major_strides(&extended_shape);

    for dim in 0..rank {
        if start[dim] + sizes[dim] > extended_shape[dim] {
            return Err(format!(
                "mat2cell: partition exceeds dimension {} bounds",
                dim + 1
            ));
        }
    }

    let total = sizes.iter().product::<usize>();
    if total == 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(total);
    let mut indices = vec![0usize; rank];
    loop {
        let mut linear = 0usize;
        for dim in 0..rank {
            linear += (start[dim] + indices[dim]) * strides[dim];
        }
        result.push(
            data.get(linear)
                .ok_or_else(|| "mat2cell: internal indexing error".to_string())?
                .clone(),
        );

        let mut carry = true;
        for dim in 0..rank {
            indices[dim] += 1;
            if indices[dim] < sizes[dim] {
                carry = false;
                break;
            }
            indices[dim] = 0;
        }
        if carry {
            break;
        }
    }
    Ok(result)
}

fn slice_char_array(array: &CharArray, start: &[usize], sizes: &[usize]) -> Result<Value, String> {
    if sizes.len() > 2 {
        for (dim, &count) in sizes.iter().enumerate().skip(2) {
            let offset = start.get(dim).copied().unwrap_or(0);
            if count != 1 || offset != 0 {
                return Err(
                    "mat2cell: character arrays cannot be partitioned along higher dimensions"
                        .to_string(),
                );
            }
        }
    }
    let row_start = start.first().copied().unwrap_or(0);
    let row_count = sizes.first().copied().unwrap_or(1);
    let col_start = start.get(1).copied().unwrap_or(0);
    let col_count = sizes.get(1).copied().unwrap_or(1);

    if row_start + row_count > array.rows || col_start + col_count > array.cols {
        return Err("mat2cell: partition exceeds character array bounds".to_string());
    }

    if row_count == 0 || col_count == 0 {
        let slice = CharArray::new(Vec::new(), row_count, col_count)
            .map_err(|e| format!("mat2cell: {e}"))?;
        return Ok(Value::CharArray(slice));
    }

    let mut data = Vec::with_capacity(row_count * col_count);
    for r in 0..row_count {
        for c in 0..col_count {
            let idx = (row_start + r) * array.cols + (col_start + c);
            data.push(array.data[idx]);
        }
    }
    let slice = CharArray::new(data, row_count, col_count).map_err(|e| format!("mat2cell: {e}"))?;
    Ok(Value::CharArray(slice))
}

fn extend_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut extended = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    while extended.len() < rank {
        extended.push(1);
    }
    extended
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1usize;
    for &extent in shape {
        strides.push(acc);
        acc = acc.saturating_mul(extent.max(1));
    }
    strides
}

fn prefix_sums(values: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(values.len());
    let mut acc = 0usize;
    for &v in values {
        result.push(acc);
        acc += v;
    }
    result
}

fn adapt_numeric_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else if shape.len() == 1 {
        vec![1, shape[0]]
    } else {
        shape.to_vec()
    }
}

fn normalize_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn adjust_output_shape(sizes: &[usize]) -> Vec<usize> {
    if sizes.is_empty() {
        vec![1, 1]
    } else {
        sizes.to_vec()
    }
}

fn normalize_cell_shape(shape: Vec<usize>) -> Vec<usize> {
    match shape.len() {
        0 => vec![0, 0],
        1 => vec![shape[0], 1],
        _ => shape,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

    fn row_vector(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![1, values.len()]).expect("row vector tensor"),
        )
    }

    fn column_vector(values: &[f64]) -> Value {
        Value::Tensor(
            Tensor::new(values.to_vec(), vec![values.len(), 1]).expect("column vector tensor"),
        )
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn partition_matrix_into_quadrants() {
        let tensor = Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let result = mat2cell_builtin(
            Value::Tensor(tensor),
            vec![row_vector(&[2.0, 2.0]), row_vector(&[1.0, 3.0])],
        )
        .expect("mat2cell");

        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![2, 2]);

        let bottom_right = (*cell.data[3]).clone();
        let gathered = test_support::gather(bottom_right).expect("gather");
        assert_eq!(gathered.shape, vec![2, 3]);
        assert_eq!(gathered.data, vec![7.0, 8.0, 11.0, 12.0, 15.0, 16.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn row_vector_with_single_partition_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]).unwrap();
        let result = mat2cell_builtin(
            Value::Tensor(tensor),
            vec![row_vector(&[1.0]), row_vector(&[2.0, 1.0, 3.0])],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![1, 3]);
        assert_eq!(cell.data.len(), 3);
        let third = (*cell.data[2]).clone();
        let gathered = test_support::gather(third).expect("gather");
        assert_eq!(gathered.data, vec![4.0, 5.0, 6.0]);
        assert_eq!(gathered.shape, vec![1, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn column_vector_with_implicit_column_partition() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = mat2cell_builtin(Value::Tensor(tensor), vec![column_vector(&[2.0, 2.0])])
            .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![2, 1]);
        let second = (*cell.data[1]).clone();
        let gathered = test_support::gather(second).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.data, vec![3.0, 4.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_partition_yields_complex_value() {
        let result = mat2cell_builtin(
            Value::Complex(1.25, -2.5),
            vec![row_vector(&[1.0]), row_vector(&[1.0])],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![1, 1]);
        let value = (*cell.data[0]).clone();
        match value {
            Value::Complex(re, im) => {
                assert!((re - 1.25).abs() < 1e-12);
                assert!((im + 2.5).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn three_dimensional_tensor_partition() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let result = mat2cell_builtin(
            Value::Tensor(tensor),
            vec![
                row_vector(&[1.0, 2.0]),
                row_vector(&[2.0, 2.0]),
                row_vector(&[1.0, 1.0]),
            ],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![2, 2, 2]);
        let index = 6; // (2,2,1) in row-major indexing
        let block = (*cell.data[index]).clone();
        let gathered = test_support::gather(block).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2, 1]);
        assert_eq!(gathered.data, vec![8.0, 9.0, 11.0, 12.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zero_sized_blocks() {
        let tensor = Tensor::new(vec![0.0; 6], vec![3, 2]).unwrap();
        let result = mat2cell_builtin(
            Value::Tensor(tensor),
            vec![row_vector(&[0.0, 3.0]), row_vector(&[1.0, 1.0])],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![2, 2]);
        let top_left = (*cell.data[0]).clone();
        let gathered = test_support::gather(top_left).expect("gather");
        assert_eq!(gathered.data.len(), 0);
        assert_eq!(gathered.shape, vec![0, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_partition_vector_supported() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let logical =
            LogicalArray::new(vec![1, 1, 1, 1], vec![4, 1]).expect("logical partition vector");
        let result = mat2cell_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(logical)])
            .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![4, 1]);
        let third = (*cell.data[2]).clone();
        let gathered = test_support::gather(third).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        assert_eq!(gathered.data, vec![3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_array_partition() {
        let chars = CharArray::new(
            vec!['f', 'o', 'o', ' ', 'b', 'a', 'r', ' ', 'b', 'a', 'z', ' '],
            3,
            4,
        )
        .unwrap();
        let result = mat2cell_builtin(
            Value::CharArray(chars),
            vec![row_vector(&[1.0, 2.0]), row_vector(&[2.0, 2.0])],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        let second = (*cell.data[1]).clone();
        match second {
            Value::CharArray(slice) => {
                assert_eq!(slice.rows, 1);
                assert_eq!(slice.cols, 2);
                let text: String = slice.data.iter().collect();
                assert_eq!(text, "o ");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatch_partition_sum_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = mat2cell_builtin(
            Value::Tensor(tensor),
            vec![row_vector(&[1.0]), row_vector(&[3.0])],
        )
        .unwrap_err();
        assert!(
            err.contains("partition sizes"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_partition_entry_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let err =
            mat2cell_builtin(Value::Tensor(tensor), vec![row_vector(&[-1.0, 5.0])]).unwrap_err();
        assert!(
            err.contains("non-negative"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_integer_partition_entry_errors() {
        let tensor = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![4, 1]).unwrap();
        let err = mat2cell_builtin(Value::Tensor(tensor), vec![row_vector(&[1.5, 0.5, 2.0])])
            .unwrap_err();
        assert!(err.contains("integers"), "unexpected error message: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mat2cell_gpu_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((1..=6).map(|v| v as f64).collect(), vec![3, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mat2cell_builtin(
                Value::GpuTensor(handle),
                vec![row_vector(&[1.0, 2.0]), row_vector(&[1.0, 1.0])],
            )
            .expect("mat2cell");
            let cell = match result {
                Value::Cell(ca) => ca,
                other => panic!("expected cell array, got {other:?}"),
            };
            assert_eq!(cell.shape, vec![2, 2]);
            let block = (*cell.data[3]).clone();
            let gathered = test_support::gather(block).expect("gather");
            assert_eq!(gathered.data, vec![5.0, 6.0]);
            assert_eq!(gathered.shape, vec![2, 1]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn mat2cell_wgpu_matches_cpu_partitions() {
        let provider = match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => provider,
            Err(_) => {
                runmat_accelerate::simple_provider::register_inprocess_provider();
                accel_provider::maybe_provider("builtins::cells::core::mat2cell::wgpu-test")
                    .expect("accel provider")
            }
        };

        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![4, 2]).unwrap();
        let cpu_result = mat2cell_builtin(
            Value::Tensor(tensor.clone()),
            vec![row_vector(&[1.0, 3.0]), row_vector(&[1.0, 1.0])],
        )
        .expect("cpu mat2cell");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_result = mat2cell_builtin(
            Value::GpuTensor(handle),
            vec![row_vector(&[1.0, 3.0]), row_vector(&[1.0, 1.0])],
        )
        .expect("gpu mat2cell");

        let cpu_cell = match cpu_result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        let gpu_cell = match gpu_result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cpu_cell.shape, gpu_cell.shape);
        assert_eq!(cpu_cell.data.len(), gpu_cell.data.len());
        for (cpu, gpu) in cpu_cell.data.iter().zip(gpu_cell.data.iter()) {
            let cpu_val = (**cpu).clone();
            let gpu_val = (**gpu).clone();
            let cpu_tensor = test_support::gather(cpu_val).expect("cpu gather");
            let gpu_tensor = test_support::gather(gpu_val).expect("gpu gather");
            assert_eq!(cpu_tensor.shape, gpu_tensor.shape);
            assert_eq!(cpu_tensor.data, gpu_tensor.data);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_exist() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
