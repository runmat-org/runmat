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
                    Tensor::new_with_dtype(data, vec![rows, 1], tensor.numeric_dtype())
                        .map_err(invalid_variable)?
                };
                out.push(Value::Tensor(value));
            }
            Ok(out)
        }
        Value::ComplexTensor(tensor) => {
            let mut out = Vec::with_capacity(tensor.cols);
            for col in 0..tensor.cols {
                let value = if let Some(storage) = &tensor.integer_storage() {
                    let mut real = Vec::with_capacity(tensor.rows);
                    let mut imag = Vec::with_capacity(tensor.rows);
                    for row in 0..tensor.rows {
                        let index = row + col * tensor.rows;
                        real.push(
                            storage
                                .real
                                .value_at(index)
                                .expect("integer tensor storage matches tensor shape"),
                        );
                        imag.push(
                            storage
                                .imag
                                .value_at(index)
                                .expect("integer tensor storage matches tensor shape"),
                        );
                    }
                    ComplexTensor::new_integer(
                        runmat_value::IntegerComplexStorage::new(
                            storage
                                .real
                                .from_exact_values_like(real)
                                .map_err(invalid_variable)?,
                            storage
                                .imag
                                .from_exact_values_like(imag)
                                .map_err(invalid_variable)?,
                        )
                        .map_err(invalid_variable)?,
                        vec![tensor.rows, 1],
                    )
                    .map_err(invalid_variable)?
                } else {
                    let mut data = Vec::with_capacity(tensor.rows);
                    for row in 0..tensor.rows {
                        data.push(tensor.materialize_f64()[row + col * tensor.rows]);
                    }
                    ComplexTensor::new(data, vec![tensor.rows, 1]).map_err(invalid_variable)?
                };
                out.push(Value::ComplexTensor(value));
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
    use runmat_value::{IntegerStorage, NumericStorage};

    #[test]
    fn split_value_columns_preserves_native_single_storage() {
        let columns = split_value_columns(Value::Tensor(
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        ))
        .unwrap();

        let Value::Tensor(first) = columns[0].clone() else {
            panic!("expected first tensor column");
        };
        let Value::Tensor(second) = columns[1].clone() else {
            panic!("expected second tensor column");
        };
        assert_eq!(
            first.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.0, 2.0])
        );
        assert_eq!(
            second.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, 4.0])
        );
    }

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

    #[test]
    fn split_value_columns_preserves_exact_complex_integer_storage() {
        let large = 9_007_199_254_740_993_i64;
        let columns = split_value_columns(Value::ComplexTensor(
            ComplexTensor::new_integer(
                runmat_value::IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![large, i64::MIN, 7, 0]),
                    IntegerStorage::I64(vec![0, 5, i64::MIN, large]),
                )
                .unwrap(),
                vec![2, 2],
            )
            .unwrap(),
        ))
        .unwrap();

        assert_eq!(columns.len(), 2);
        let Value::ComplexTensor(first) = &columns[0] else {
            panic!("expected first complex tensor column");
        };
        let Value::ComplexTensor(second) = &columns[1] else {
            panic!("expected second complex tensor column");
        };
        assert_eq!(
            first.integer_storage().cloned(),
            Some(
                runmat_value::IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![large, i64::MIN]),
                    IntegerStorage::I64(vec![0, 5]),
                )
                .unwrap(),
            )
        );
        assert_eq!(
            second.integer_storage().cloned(),
            Some(
                runmat_value::IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![7, 0]),
                    IntegerStorage::I64(vec![i64::MIN, large]),
                )
                .unwrap(),
            )
        );
    }
}
