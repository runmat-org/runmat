//! Matrix indexing and slicing operations
//!
//! Implements language-style tensor indexing and access patterns.

use crate::builtins::common::shape::normalize_scalar_shape;
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{ComplexTensor, IntegerComplexStorage, SparseTensor, Tensor, Value};

fn indexing_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).build()
}

fn indexing_error_with_identifier(message: impl Into<String>, identifier: &str) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier)
        .build()
}

fn positive_integer_cell_index(value: f64, identifier: &str) -> Result<usize, RuntimeError> {
    let Some(index) = positive_platform_index(value) else {
        return Err(indexing_error_with_identifier(
            format!("Cell index {value} must be a positive integer"),
            identifier,
        ));
    };
    Ok(index)
}

fn positive_integer_index(value: f64, identifier: &str) -> Result<usize, RuntimeError> {
    let Some(index) = positive_platform_index(value) else {
        return Err(indexing_error_with_identifier(
            format!("Index {value} must be a positive integer"),
            identifier,
        ));
    };
    Ok(index)
}

fn positive_platform_index(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn tensor_scalar_value(tensor: &Tensor, index: usize) -> Result<Value, RuntimeError> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage.value_at(index).ok_or_else(|| {
            indexing_error_with_identifier(
                "Integer scalar index out of bounds",
                "RunMat:IndexOutOfBounds",
            )
        })?;
        return Ok(Value::Int(value));
    }
    tensor
        .data
        .get(index)
        .copied()
        .map(Value::Num)
        .ok_or_else(|| {
            indexing_error_with_identifier(
                "Tensor scalar index out of bounds",
                "RunMat:IndexOutOfBounds",
            )
        })
}

fn complex_tensor_scalar_value(
    tensor: &ComplexTensor,
    index: usize,
) -> Result<Value, RuntimeError> {
    if let Some(storage) = &tensor.integer_data {
        let real = storage.real.value_at(index).ok_or_else(|| {
            indexing_error_with_identifier(
                "Complex integer scalar index out of bounds",
                "RunMat:IndexOutOfBounds",
            )
        })?;
        let imag = storage.imag.value_at(index).ok_or_else(|| {
            indexing_error_with_identifier(
                "Complex integer scalar index out of bounds",
                "RunMat:IndexOutOfBounds",
            )
        })?;
        let real = storage
            .real
            .from_same_class_values(vec![real])
            .map_err(indexing_error)?;
        let imag = storage
            .imag
            .from_same_class_values(vec![imag])
            .map_err(indexing_error)?;
        let scalar = ComplexTensor::new_integer(
            IntegerComplexStorage::new(real, imag).map_err(indexing_error)?,
            vec![1, 1],
        )
        .map_err(indexing_error)?;
        return Ok(Value::ComplexTensor(scalar));
    }
    tensor
        .data
        .get(index)
        .copied()
        .map(|(re, im)| Value::Complex(re, im))
        .ok_or_else(|| {
            indexing_error_with_identifier(
                "Complex scalar index out of bounds",
                "RunMat:IndexOutOfBounds",
            )
        })
}

fn sparse_scalar_index(sparse: &SparseTensor, indices: &[f64]) -> Result<Value, RuntimeError> {
    fn sparse_scalar(sparse: &SparseTensor, row: usize, col: usize) -> Result<Value, RuntimeError> {
        if let Some(storage) = sparse.integer_storage() {
            let result = match sparse.integer_at(row, col) {
                Some(value) => {
                    SparseTensor::new_integer_like(1, 1, vec![0, 1], vec![0], vec![value], storage)
                }
                None => Ok(SparseTensor::zeros_with_integer_storage(1, 1, storage)),
            }
            .map_err(indexing_error)?;
            return Ok(Value::SparseTensor(result));
        }

        let value = sparse.get(row, col).unwrap_or(0.0);
        if value == 0.0 {
            return Ok(Value::SparseTensor(SparseTensor::zeros(1, 1)));
        }
        Ok(Value::SparseTensor(
            SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![value]).map_err(indexing_error)?,
        ))
    }

    if indices.is_empty() {
        return Err(indexing_error("At least one index is required"));
    }
    if indices.len() == 1 {
        let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
        let total = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
            indexing_error_with_identifier("Sparse dimensions overflow", "RunMat:IndexOutOfBounds")
        })?;
        if idx > total {
            return Err(indexing_error_with_identifier(
                format!("Index {idx} out of bounds (1 to {total})"),
                "RunMat:IndexOutOfBounds",
            ));
        }
        let zero = idx - 1;
        let row = zero % sparse.rows;
        let col = zero / sparse.rows;
        return sparse_scalar(sparse, row, col);
    }
    if indices.len() == 2 {
        let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
        let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
        if row > sparse.rows || col > sparse.cols {
            return Err(indexing_error_with_identifier(
                format!(
                    "Index ({row}, {col}) out of bounds for {}x{} sparse matrix",
                    sparse.rows, sparse.cols
                ),
                "RunMat:IndexOutOfBounds",
            ));
        }
        return sparse_scalar(sparse, row - 1, col - 1);
    }
    Err(indexing_error_with_identifier(
        format!(
            "Sparse matrices support 1 or 2 indices, got {}",
            indices.len()
        ),
        "RunMat:SliceNonTensor",
    ))
}

fn cell_row_major_pos_from_linear(
    ca: &runmat_builtins::CellArray,
    idx: usize,
) -> Result<usize, RuntimeError> {
    if idx == 0 || idx > ca.data.len() {
        return Err(indexing_error_with_identifier(
            format!("Cell index {} out of bounds (1 to {})", idx, ca.data.len()),
            "RunMat:CellIndexOutOfBounds",
        ));
    }
    if ca.rows <= 1 || ca.cols <= 1 {
        return Ok(idx - 1);
    }
    let zero = idx - 1;
    let row = zero % ca.rows;
    let col = zero / ca.rows;
    Ok(row * ca.cols + col)
}

/// Get a single element from a tensor (1-based indexing like language)
pub fn matrix_get_element(tensor: &Tensor, row: usize, col: usize) -> Result<f64, RuntimeError> {
    if row == 0 || col == 0 {
        return Err(indexing_error_with_identifier(
            "MATLAB uses 1-based indexing",
            "RunMat:IndexOutOfBounds",
        ));
    }
    tensor
        .get2(row - 1, col - 1)
        .map_err(|err| indexing_error_with_identifier(err, "RunMat:IndexOutOfBounds"))
}

/// Set a single element in a tensor (1-based indexing like language)
pub fn matrix_set_element(
    tensor: &mut Tensor,
    row: usize,
    col: usize,
    value: f64,
) -> Result<(), RuntimeError> {
    if row == 0 || col == 0 {
        return Err(indexing_error_with_identifier(
            "The MATLAB language uses 1-based indexing",
            "RunMat:IndexOutOfBounds",
        ));
    }
    let index = (row - 1) + (col - 1) * tensor.rows();
    let mut compatibility_value = value;
    if let Some(storage) = &mut tensor.integer_data {
        let exact = storage.cast_f64_assignment(value);
        compatibility_value = exact.to_f64();
        storage
            .set_value(index, exact)
            .map_err(|err| indexing_error_with_identifier(err, "RunMat:IndexOutOfBounds"))?;
    }
    tensor
        .set2(row - 1, col - 1, compatibility_value)
        .map_err(|err| indexing_error_with_identifier(err, "RunMat:IndexOutOfBounds"))
}

/// Get a row from a tensor
pub fn matrix_get_row(tensor: &Tensor, row: usize) -> Result<Tensor, RuntimeError> {
    if row == 0 || row > tensor.rows() {
        return Err(indexing_error_with_identifier(
            format!(
                "Row index {} out of bounds for {}x{} tensor",
                row,
                tensor.rows(),
                tensor.cols()
            ),
            "RunMat:IndexOutOfBounds",
        ));
    }

    if let Some(storage) = tensor.integer_storage() {
        let values = (0..tensor.cols())
            .map(|col| storage.value_at((row - 1) + col * tensor.rows()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| indexing_error("integer storage is inconsistent"))?;
        return Tensor::new_integer(
            storage
                .from_same_class_values(values)
                .map_err(indexing_error)?,
            vec![1, tensor.cols()],
        )
        .map_err(indexing_error);
    }

    // Column-major: row slice picks every element spaced by rows across columns
    let mut row_data = Vec::with_capacity(tensor.cols());
    for c in 0..tensor.cols() {
        row_data.push(tensor.data[(row - 1) + c * tensor.rows()]);
    }
    Tensor::new_2d(row_data, 1, tensor.cols()).map_err(|err| indexing_error(err))
}

/// Get a column from a tensor
pub fn matrix_get_col(tensor: &Tensor, col: usize) -> Result<Tensor, RuntimeError> {
    if col == 0 || col > tensor.cols() {
        return Err(indexing_error_with_identifier(
            format!(
                "Column index {} out of bounds for {}x{} tensor",
                col,
                tensor.rows(),
                tensor.cols()
            ),
            "RunMat:IndexOutOfBounds",
        ));
    }

    if let Some(storage) = tensor.integer_storage() {
        let start = (col - 1) * tensor.rows();
        let values = (0..tensor.rows())
            .map(|row| storage.value_at(start + row))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| indexing_error("integer storage is inconsistent"))?;
        return Tensor::new_integer(
            storage
                .from_same_class_values(values)
                .map_err(indexing_error)?,
            vec![tensor.rows(), 1],
        )
        .map_err(indexing_error);
    }

    let mut col_data = Vec::with_capacity(tensor.rows());
    for row in 0..tensor.rows() {
        col_data.push(tensor.data[row + (col - 1) * tensor.rows()]);
    }
    Tensor::new_2d(col_data, tensor.rows(), 1).map_err(|err| indexing_error(err))
}

/// Array indexing operation (used by all interpreters/compilers)
/// In MATLAB, indexing is 1-based and supports:
/// - Single element: A(i) for vectors, A(i,j) for tensors
/// - Multiple indices: A(i1, i2, ..., iN)
pub async fn perform_indexing(base: &Value, indices: &[f64]) -> Result<Value, RuntimeError> {
    match base {
        Value::GpuTensor(h) => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                indexing_error("Cannot index value of type GpuTensor without a provider")
            })?;
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            // Support scalar indexing cases mirroring Tensor branch
            if indices.len() == 1 {
                let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let total = h.shape.iter().product();
                if idx < 1 || idx > total {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, total),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                let lin0 = idx - 1; // 0-based
                let val = gpu_index_scalar(provider, h, lin0).await?;
                return Ok(Value::Num(val));
            } else if indices.len() == 2 {
                let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
                let rows = h.shape.first().copied().unwrap_or(1);
                let cols = h.shape.get(1).copied().unwrap_or(1);
                if row < 1 || row > rows || col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        format!("Index ({row}, {col}) out of bounds for {rows}x{cols} tensor"),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                let lin0 = (row - 1) + (col - 1) * rows;
                let val = gpu_index_scalar(provider, h, lin0).await?;
                return Ok(Value::Num(val));
            }
            Err(indexing_error_with_identifier(
                format!("Cannot index value of type {base:?}"),
                "RunMat:SliceNonTensor",
            ))
        }
        Value::Tensor(tensor) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }

            if indices.len() == 1 {
                // Linear indexing (1-based)
                let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let len = tensor_utils::tensor_element_len(tensor);
                if idx < 1 || idx > len {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, len),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                tensor_scalar_value(tensor, idx - 1)
            } else if indices.len() == 2 {
                // Row-column indexing (1-based)
                let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
                let shape = normalize_scalar_shape(&tensor.shape);
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);

                if row < 1 || row > rows {
                    return Err(indexing_error_with_identifier(
                        format!("Row index {} out of bounds (1 to {})", row, rows),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                if col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        format!("Column index {} out of bounds (1 to {})", col, cols),
                        "RunMat:IndexOutOfBounds",
                    ));
                }

                let linear_idx = (row - 1) + (col - 1) * rows; // Convert to 0-based, column-major
                tensor_scalar_value(tensor, linear_idx)
            } else {
                Err(indexing_error(format!(
                    "Tensors support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::LogicalArray(array) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }

            if indices.len() == 1 {
                let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                if idx < 1 || idx > array.data.len() {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, array.data.len()),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                Ok(Value::Bool(array.data[idx - 1] != 0))
            } else if indices.len() == 2 {
                let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
                let shape = normalize_scalar_shape(&array.shape);
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);

                if row < 1 || row > rows {
                    return Err(indexing_error_with_identifier(
                        format!("Row index {} out of bounds (1 to {})", row, rows),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                if col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        format!("Column index {} out of bounds (1 to {})", col, cols),
                        "RunMat:IndexOutOfBounds",
                    ));
                }

                let linear_idx = (row - 1) + (col - 1) * rows;
                Ok(Value::Bool(array.data[linear_idx] != 0))
            } else {
                Err(indexing_error(format!(
                    "Logical arrays support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::ComplexTensor(tensor) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }

            if indices.len() == 1 {
                let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                if idx < 1 || idx > tensor.data.len() {
                    return Err(indexing_error_with_identifier(
                        format!("Index {} out of bounds (1 to {})", idx, tensor.data.len()),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                complex_tensor_scalar_value(tensor, idx - 1)
            } else if indices.len() == 2 {
                let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
                let shape = normalize_scalar_shape(&tensor.shape);
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);

                if row < 1 || row > rows {
                    return Err(indexing_error_with_identifier(
                        format!("Row index {} out of bounds (1 to {})", row, rows),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                if col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        format!("Column index {} out of bounds (1 to {})", col, cols),
                        "RunMat:IndexOutOfBounds",
                    ));
                }

                let linear_idx = (row - 1) + (col - 1) * rows;
                complex_tensor_scalar_value(tensor, linear_idx)
            } else {
                Err(indexing_error(format!(
                    "Complex tensors support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::SparseTensor(sparse) => sparse_scalar_index(sparse, indices),
        Value::StringArray(sa) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            if indices.len() == 1 {
                let idx = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let total = sa.data.len();
                if idx < 1 || idx > total {
                    return Err(indexing_error_with_identifier(
                        format!("Index {idx} out of bounds (1 to {total})"),
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                Ok(Value::String(sa.data[idx - 1].clone()))
            } else if indices.len() == 2 {
                let row = positive_integer_index(indices[0], "RunMat:IndexOutOfBounds")?;
                let col = positive_integer_index(indices[1], "RunMat:IndexOutOfBounds")?;
                let shape = normalize_scalar_shape(&sa.shape);
                let rows = shape.first().copied().unwrap_or(1);
                let cols = shape.get(1).copied().unwrap_or(1);
                if row < 1 || row > rows || col < 1 || col > cols {
                    return Err(indexing_error_with_identifier(
                        "StringArray subscript out of bounds",
                        "RunMat:IndexOutOfBounds",
                    ));
                }
                let idx = (row - 1) + (col - 1) * rows;
                Ok(Value::String(sa.data[idx].clone()))
            } else {
                Err(indexing_error(format!(
                    "StringArray supports 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::Num(_) | Value::Int(_) => {
            if indices.len() == 1 && indices[0] == 1.0 {
                // Scalar indexing with A(1) returns the scalar itself
                Ok(base.clone())
            } else {
                Err(indexing_error_with_identifier(
                    "Slicing only supported on tensors",
                    "RunMat:SliceNonTensor",
                ))
            }
        }
        Value::Cell(ca) => {
            if indices.is_empty() {
                return Err(indexing_error("At least one index is required"));
            }
            if indices.len() == 1 {
                let idx = positive_integer_cell_index(indices[0], "RunMat:CellIndexOutOfBounds")?;
                if idx < 1 || idx > ca.data.len() {
                    return Err(indexing_error_with_identifier(
                        format!("Cell index {} out of bounds (1 to {})", idx, ca.data.len()),
                        "RunMat:CellIndexOutOfBounds",
                    ));
                }
                let pos = cell_row_major_pos_from_linear(ca, idx)?;
                Ok(ca.data[pos].clone())
            } else if indices.len() == 2 {
                let row =
                    positive_integer_cell_index(indices[0], "RunMat:CellSubscriptOutOfBounds")?;
                let col =
                    positive_integer_cell_index(indices[1], "RunMat:CellSubscriptOutOfBounds")?;
                if row < 1 || row > ca.rows || col < 1 || col > ca.cols {
                    return Err(indexing_error_with_identifier(
                        "Cell subscript out of bounds",
                        "RunMat:CellSubscriptOutOfBounds",
                    ));
                }
                Ok(ca.data[(row - 1) * ca.cols + (col - 1)].clone())
            } else {
                Err(indexing_error(format!(
                    "Cell arrays support 1 or 2 indices, got {}",
                    indices.len()
                )))
            }
        }
        Value::Struct(_) => {
            if matches!(indices, [1.0] | [1.0, 1.0]) {
                Ok(base.clone())
            } else {
                Err(indexing_error_with_identifier(
                    "Struct subscript out of bounds",
                    "RunMat:IndexOutOfBounds",
                ))
            }
        }
        _ => Err(indexing_error_with_identifier(
            format!("Cannot index value of type {base:?}"),
            "RunMat:SliceNonTensor",
        )),
    }
}

async fn gpu_index_scalar(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    handle: &runmat_accelerate_api::GpuTensorHandle,
    lin0: usize,
) -> Result<f64, RuntimeError> {
    #[cfg(target_arch = "wasm32")]
    {
        let host = provider
            .download(handle)
            .await
            .map_err(|e| indexing_error(format!("gpu index: {e}")))?;
        if lin0 >= host.data.len() {
            return Err(indexing_error(format!(
                "gpu index: index {} out of bounds (len {})",
                lin0 + 1,
                host.data.len()
            )));
        }
        Ok(host.data[lin0])
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        provider
            .read_scalar(handle, lin0)
            .map_err(|e| indexing_error(format!("gpu index: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::{matrix_get_col, matrix_get_row, matrix_set_element, perform_indexing};
    use futures::executor::block_on;
    use runmat_builtins::{
        CellArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
        SparseTensor, StringArray, Tensor, Value,
    };

    fn sparse_scalar_value(value: Value) -> f64 {
        match value {
            Value::SparseTensor(sparse) if sparse.shape() == vec![1, 1] => {
                sparse.get(0, 0).unwrap_or(0.0)
            }
            other => panic!("expected sparse scalar value, got {other:?}"),
        }
    }

    #[test]
    fn cell_index_rejects_fractional_before_cast() {
        let cell = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            1,
            4,
        )
        .expect("cell");
        let err = block_on(perform_indexing(&Value::Cell(cell), &[3.7]))
            .expect_err("fractional cell index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
    }

    #[test]
    fn host_indexing_rejects_fractional_and_out_of_range_selectors_before_cast() {
        let values = [
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical")),
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2])
                    .expect("complex tensor"),
            ),
            Value::StringArray(
                StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("string array"),
            ),
        ];

        let out_of_range = usize::MAX as f64 + 1.0;
        for value in values {
            for selector in [1.5, out_of_range] {
                let err = block_on(perform_indexing(&value, &[selector]))
                    .expect_err("invalid selector must fail before cast");
                assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
            }
        }

        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = block_on(perform_indexing(&Value::Cell(cell), &[out_of_range]))
            .expect_err("out-of-range cell selector must fail before cast");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexOutOfBounds"));
    }

    #[test]
    fn sparse_indexing_reads_stored_unstored_and_linear_values() {
        let sparse = SparseTensor::new(
            3,
            3,
            vec![0, 2, 2, 3],
            vec![0, 2, 1],
            vec![10.0, 30.0, 23.0],
        )
        .unwrap();

        assert_eq!(
            sparse_scalar_value(
                block_on(perform_indexing(
                    &Value::SparseTensor(sparse.clone()),
                    &[1.0, 1.0]
                ))
                .unwrap()
            ),
            10.0
        );
        assert_eq!(
            sparse_scalar_value(
                block_on(perform_indexing(
                    &Value::SparseTensor(sparse.clone()),
                    &[2.0, 1.0]
                ))
                .unwrap()
            ),
            0.0
        );
        assert_eq!(
            sparse_scalar_value(
                block_on(perform_indexing(&Value::SparseTensor(sparse), &[8.0])).unwrap()
            ),
            23.0
        );
    }

    #[test]
    fn sparse_integer_scalar_indexing_preserves_exact_class_and_zero() {
        let cases = vec![
            IntegerStorage::I8(vec![i8::MIN]),
            IntegerStorage::I16(vec![i16::MIN]),
            IntegerStorage::I32(vec![i32::MIN]),
            IntegerStorage::I64(vec![i64::MIN]),
            IntegerStorage::U8(vec![u8::MAX]),
            IntegerStorage::U16(vec![u16::MAX]),
            IntegerStorage::U32(vec![u32::MAX]),
            IntegerStorage::U64(vec![u64::MAX]),
        ];

        for storage in cases {
            let sparse = SparseTensor::new_integer(2, 2, vec![0, 1, 1], vec![0], storage.clone())
                .expect("typed sparse");
            for indices in [&[1.0][..], &[1.0, 1.0][..], &[2.0, 2.0][..]] {
                let result = block_on(perform_indexing(
                    &Value::SparseTensor(sparse.clone()),
                    indices,
                ))
                .expect("sparse scalar index");
                let Value::SparseTensor(result) = result else {
                    panic!("expected sparse scalar result");
                };
                assert_eq!(
                    result.integer_storage().map(IntegerStorage::class_name),
                    Some(storage.class_name())
                );
                let expected = if indices == &[1.0] || indices == &[1.0, 1.0] {
                    storage.clone()
                } else {
                    storage.zeros_like(0)
                };
                assert_eq!(result.integer_storage(), Some(&expected));
            }
        }
    }

    #[test]
    fn dense_integer_scalar_indexing_preserves_the_exact_value() {
        let mut tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
            vec![1, 2],
        )
        .expect("tensor");
        tensor.data.clear();

        assert_eq!(
            block_on(perform_indexing(&Value::Tensor(tensor), &[2.0])).expect("scalar index"),
            Value::Int(IntValue::U64(u64::MAX))
        );
    }

    #[test]
    fn typed_complex_integer_scalar_indexing_preserves_exact_components() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
                IntegerStorage::U64(vec![7, 8]),
            )
            .expect("storage"),
            vec![1, 2],
        )
        .expect("tensor");

        let result = block_on(perform_indexing(&Value::ComplexTensor(tensor), &[2.0]))
            .expect("scalar index");
        let Value::ComplexTensor(result) = result else {
            panic!("typed complex integer scalar must retain exact storage");
        };
        assert_eq!(
            result.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![8]),
                )
                .expect("expected storage")
            )
        );
    }

    #[test]
    fn common_tensor_index_helpers_preserve_exact_integer_storage() {
        let large = 9_007_199_254_740_993_u64;
        let mut tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX, 3, 4]), vec![2, 2])
                .expect("tensor");

        assert_eq!(
            matrix_get_row(&tensor, 2).unwrap().integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 4]))
        );
        assert_eq!(
            matrix_get_col(&tensor, 1).unwrap().integer_storage(),
            Some(&IntegerStorage::U64(vec![large, u64::MAX]))
        );

        matrix_set_element(&mut tensor, 1, 2, -4.2).unwrap();
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, u64::MAX, 0, 4]))
        );
        assert_eq!(tensor.data[2], 0.0);
    }

    #[test]
    fn logical_indexing_reads_linear_and_row_column_scalars() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical");

        assert_eq!(
            block_on(perform_indexing(
                &Value::LogicalArray(logical.clone()),
                &[1.0]
            ))
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            block_on(perform_indexing(
                &Value::LogicalArray(logical.clone()),
                &[2.0, 1.0]
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            block_on(perform_indexing(&Value::LogicalArray(logical), &[2.0, 2.0])).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn cell_subscript_rejects_nan_before_cast() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = block_on(perform_indexing(&Value::Cell(cell), &[f64::NAN, 1.0]))
            .expect_err("NaN cell subscript should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellSubscriptOutOfBounds"));
    }

    #[test]
    fn cell_linear_index_uses_column_major_semantics_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::String("r1c1".to_string()),
                Value::String("r1c2".to_string()),
                Value::String("r2c1".to_string()),
                Value::String("r2c2".to_string()),
            ],
            2,
            2,
        )
        .expect("cell");
        let value = block_on(perform_indexing(&Value::Cell(cell), &[2.0])).expect("cell read");
        assert_eq!(value, Value::String("r2c1".to_string()));
    }
}
