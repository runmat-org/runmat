//! MATLAB-compatible `mat2cell` builtin for RunMat.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

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

const IDENT_INVALID_INPUT: &str = "MATLAB:mat2cell:InvalidInput";
const IDENT_INVALID_PARTITION: &str = "MATLAB:mat2cell:InvalidPartition";
const IDENT_SIZE_LIMIT: &str = "MATLAB:mat2cell:SizeExceeded";

fn mat2cell_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("mat2cell")
        .build()
}

fn mat2cell_error_with_identifier(message: impl Into<String>, identifier: &str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("mat2cell")
        .with_identifier(identifier)
        .build()
}

#[runtime_builtin(
    name = "mat2cell",
    category = "cells/core",
    summary = "Split arrays into cell-array blocks.",
    keywords = "mat2cell,cell array,partition,block",
    builtin_path = "crate::builtins::cells::core::mat2cell"
)]
async fn mat2cell_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(mat2cell_error_with_identifier(
            "mat2cell: expected at least one size vector",
            IDENT_INVALID_INPUT,
        ));
    }

    let host_value = gather_if_needed_async(&value).await?;
    let mut size_args = Vec::with_capacity(rest.len());
    for arg in rest {
        let gathered = gather_if_needed_async(&arg).await?;
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
    fn try_new(value: Value) -> BuiltinResult<Self> {
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
                    StringArray::new(vec![s], vec![1, 1]).map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
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
                    .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
                Self::try_new(Value::ComplexTensor(tensor))
            }
            Value::Num(n) => {
                let tensor = tensor::value_into_tensor_for("mat2cell", Value::Num(n))
                    .map_err(mat2cell_error)?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            Value::Int(i) => {
                let tensor = tensor::value_into_tensor_for("mat2cell", Value::Int(i.clone()))
                    .map_err(mat2cell_error)?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            Value::Bool(b) => {
                let tensor = tensor::value_into_tensor_for("mat2cell", Value::Bool(b))
                    .map_err(mat2cell_error)?;
                Mat2CellInput::try_new(Value::Tensor(tensor))
            }
            other => Err(mat2cell_error_with_identifier(
                format!(
                    "mat2cell: unsupported input type {other:?}; expected numeric, logical, string, or char arrays"
                ),
                IDENT_INVALID_INPUT,
            )),
        }
    }

    fn normalized_dims(&self) -> &[usize] {
        &self.normalized_dims
    }

    fn extract(&self, start: &[usize], sizes: &[usize]) -> BuiltinResult<Value> {
        match &self.kind {
            Mat2CellKind::Tensor(t) => {
                let data = copy_block(&t.data, &self.base_shape, start, sizes)?;
                let shape = adjust_output_shape(sizes);
                let tensor = Tensor::new(data, shape)
                    .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
                Ok(tensor::tensor_into_value(tensor))
            }
            Mat2CellKind::Complex(t) => {
                let data = copy_block(&t.data, &self.base_shape, start, sizes)?;
                let shape = adjust_output_shape(sizes);
                if data.len() == 1 {
                    let (re, im) = data[0];
                    Ok(Value::Complex(re, im))
                } else {
                    let tensor = ComplexTensor::new(data, shape)
                        .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
                    Ok(Value::ComplexTensor(tensor))
                }
            }
            Mat2CellKind::Logical(arr) => {
                let data = copy_block(&arr.data, &self.base_shape, start, sizes)?;
                if data.len() == 1 {
                    Ok(Value::Bool(data[0] != 0))
                } else {
                    let shape = adjust_output_shape(sizes);
                    let logical = LogicalArray::new(data, shape)
                        .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
                    Ok(Value::LogicalArray(logical))
                }
            }
            Mat2CellKind::Strings(arr) => {
                let data = copy_block(&arr.data, &self.base_shape, start, sizes)?;
                if data.len() == 1 {
                    Ok(Value::String(data.into_iter().next().unwrap()))
                } else {
                    let shape = adjust_output_shape(sizes);
                    let strings = StringArray::new(data, shape)
                        .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
                    Ok(Value::StringArray(strings))
                }
            }
            Mat2CellKind::Char(ca) => slice_char_array(ca, start, sizes),
        }
    }
}

fn parse_partitions(dims: &[usize], size_args: &[Value]) -> BuiltinResult<Vec<Vec<usize>>> {
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

fn split_into_cells(input: &Mat2CellInput, partitions: Vec<Vec<usize>>) -> BuiltinResult<Value> {
    let mut per_dim_counts: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
    if per_dim_counts.is_empty() {
        per_dim_counts = vec![1, 1];
    }
    let normalized_shape = normalize_cell_shape(per_dim_counts.clone());
    let total_cells = normalized_shape.iter().product::<usize>();

    if total_cells == 0 || partitions.iter().any(|p| p.is_empty()) {
        return make_cell_with_shape(Vec::new(), normalized_shape)
            .map_err(|e| mat2cell_error(format!("mat2cell: {e}")));
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

    make_cell_with_shape(cells, normalized_shape)
        .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))
}

fn parse_partition_vector(
    value: &Value,
    dim_size: usize,
    dim_index: usize,
) -> BuiltinResult<Vec<usize>> {
    let numbers = extract_numeric_vector(value).ok_or_else(|| {
        mat2cell_error_with_identifier(
            format!(
                "mat2cell: size arguments must be numeric for dimension {}",
                dim_index
            ),
            IDENT_INVALID_PARTITION,
        )
    })?;

    if numbers.is_empty() {
        if dim_size == 0 {
            return Ok(Vec::new());
        }
        return Err(mat2cell_error_with_identifier(
            format!(
                "mat2cell: partition sizes for dimension {} must sum to {}",
                dim_index, dim_size
            ),
            IDENT_INVALID_PARTITION,
        ));
    }

    let mut total: usize = 0;
    let mut parts = Vec::with_capacity(numbers.len());
    for (idx, n) in numbers.iter().enumerate() {
        if !n.is_finite() {
            return Err(mat2cell_error_with_identifier(
                format!(
                    "mat2cell: size entries must be finite (dimension {}, index {})",
                    dim_index,
                    idx + 1
                ),
                IDENT_INVALID_PARTITION,
            ));
        }
        let rounded = n.round();
        if (rounded - n).abs() > f64::EPSILON {
            return Err(mat2cell_error_with_identifier(
                format!(
                    "mat2cell: size entries must be integers (dimension {}, index {})",
                    dim_index,
                    idx + 1
                ),
                IDENT_INVALID_PARTITION,
            ));
        }
        if rounded < 0.0 {
            return Err(mat2cell_error_with_identifier(
                format!(
                    "mat2cell: size entries must be non-negative (dimension {}, index {})",
                    dim_index,
                    idx + 1
                ),
                IDENT_INVALID_PARTITION,
            ));
        }
        let value = rounded as usize;
        total = total.checked_add(value).ok_or_else(|| {
            mat2cell_error_with_identifier(
                "mat2cell: partition sum exceeds platform limits",
                IDENT_SIZE_LIMIT,
            )
        })?;
        parts.push(value);
    }

    if total != dim_size {
        return Err(mat2cell_error_with_identifier(
            format!(
                "mat2cell: partition sizes for dimension {} must sum to {} (got {})",
                dim_index, dim_size, total
            ),
            IDENT_INVALID_PARTITION,
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
) -> BuiltinResult<Vec<T>> {
    let rank = sizes.len();
    let extended_shape = extend_shape(shape, rank);
    let strides = column_major_strides(&extended_shape);

    for dim in 0..rank {
        if start[dim] + sizes[dim] > extended_shape[dim] {
            return Err(mat2cell_error_with_identifier(
                format!("mat2cell: partition exceeds dimension {} bounds", dim + 1),
                IDENT_INVALID_PARTITION,
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
                .ok_or_else(|| {
                    mat2cell_error_with_identifier(
                        "mat2cell: internal indexing error",
                        IDENT_INVALID_PARTITION,
                    )
                })?
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

fn slice_char_array(array: &CharArray, start: &[usize], sizes: &[usize]) -> BuiltinResult<Value> {
    if sizes.len() > 2 {
        for (dim, &count) in sizes.iter().enumerate().skip(2) {
            let offset = start.get(dim).copied().unwrap_or(0);
            if count != 1 || offset != 0 {
                return Err(mat2cell_error_with_identifier(
                    "mat2cell: character arrays cannot be partitioned along higher dimensions",
                    IDENT_INVALID_PARTITION,
                ));
            }
        }
    }
    let row_start = start.first().copied().unwrap_or(0);
    let row_count = sizes.first().copied().unwrap_or(1);
    let col_start = start.get(1).copied().unwrap_or(0);
    let col_count = sizes.get(1).copied().unwrap_or(1);

    if row_start + row_count > array.rows || col_start + col_count > array.cols {
        return Err(mat2cell_error_with_identifier(
            "mat2cell: partition exceeds character array bounds",
            IDENT_INVALID_PARTITION,
        ));
    }

    if row_count == 0 || col_count == 0 {
        let slice = CharArray::new(Vec::new(), row_count, col_count)
            .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
        return Ok(Value::CharArray(slice));
    }

    let mut data = Vec::with_capacity(row_count * col_count);
    for r in 0..row_count {
        for c in 0..col_count {
            let idx = (row_start + r) * array.cols + (col_start + c);
            data.push(array.data[idx]);
        }
    }
    let slice = CharArray::new(data, row_count, col_count)
        .map_err(|e| mat2cell_error(format!("mat2cell: {e}")))?;
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
    use futures::executor::block_on;
    use runmat_builtins::LogicalArray;

    fn mat2cell_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::mat2cell_builtin(value, rest))
    }

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
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("partition sizes"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_partition_entry_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let err = mat2cell_builtin(Value::Tensor(tensor), vec![row_vector(&[-1.0, 5.0])])
            .unwrap_err()
            .to_string();
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
            .unwrap_err()
            .to_string();
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
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

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
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
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
}
