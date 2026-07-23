use super::*;

pub(in crate::builtins::table) fn split_value_columns(value: Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let value = if let Some(storage) = tensor.integer_storage() {
                    let values = (0..rows)
                        .map(|row| {
                            storage
                                .value_at(row + col * rows)
                                .expect("integer tensor storage matches tensor shape")
                        })
                        .collect();
                    Tensor::new_integer(
                        storage
                            .from_exact_values_like(values)
                            .map_err(invalid_variable)?,
                        vec![rows, 1],
                    )
                    .map_err(invalid_variable)?
                } else {
                    let mut data = Vec::with_capacity(rows);
                    for row in 0..rows {
                        data.push(tensor.get2(row, col).map_err(invalid_index)?);
                    }
                    Tensor::new_with_dtype(data, vec![rows, 1], tensor.dtype)
                        .map_err(invalid_variable)?
                };
                out.push(Value::Tensor(value));
            }
            Ok(out)
        }
        Value::ComplexTensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.cols);
            for col in 0..tensor.cols {
                let mut data = Vec::with_capacity(tensor.rows);
                for row in 0..tensor.rows {
                    data.push(tensor.data[row + col * tensor.rows]);
                }
                out.push(Value::ComplexTensor(
                    ComplexTensor::new(data, vec![tensor.rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::StringArray(array) => {
            let rows = array.rows();
            let cols = array.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(array.data[row + col * rows].clone());
                }
                out.push(Value::StringArray(
                    StringArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::LogicalArray(array) => {
            let rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(*array.data.get(row + col * rows).ok_or_else(|| {
                        invalid_variable("array2table: logical array shape mismatch")
                    })?);
                }
                out.push(Value::LogicalArray(
                    LogicalArray::new(data, vec![rows, 1]).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.cols);
            for col in 0..cell.cols {
                let mut data = Vec::with_capacity(cell.rows);
                for row in 0..cell.rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Cell(
                    CellArray::new(data, cell.rows, 1).map_err(invalid_variable)?,
                ));
            }
            Ok(out)
        }
        other => Ok(vec![other]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn split_value_columns_preserves_exact_integer_storage() {
        let large = 9_007_199_254_740_993_u64;
        let columns = split_value_columns(Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX, 7, 0]), vec![2, 2])
                .unwrap(),
        ))
        .unwrap();

        assert_eq!(columns.len(), 2);
        let Value::Tensor(first) = &columns[0] else {
            panic!("expected first tensor column");
        };
        let Value::Tensor(second) = &columns[1] else {
            panic!("expected second tensor column");
        };
        assert_eq!(
            first.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, u64::MAX]))
        );
        assert_eq!(
            second.integer_storage(),
            Some(&IntegerStorage::U64(vec![7, 0]))
        );
    }
}
