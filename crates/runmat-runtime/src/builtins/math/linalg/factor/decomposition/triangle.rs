use super::*;

pub(super) fn keep_triangle(matrix: Value, flag: TriangularFlag) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(mut tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        tensor.set2(row, col, 0.0).map_err(internal)?;
                    }
                }
            }
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(mut array) => {
            let rows = logical_rows(&array);
            let cols = logical_cols(&array);
            for col in 0..cols {
                for row in 0..rows {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        array.data[row + col * rows] = 0;
                    }
                }
            }
            Ok(Value::LogicalArray(array))
        }
        Value::ComplexTensor(mut tensor) if tensor.integer_storage().is_none() => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        tensor
                            .set_f64_assignment_at(row + col * tensor.rows(), 0.0, 0.0)
                            .map_err(internal)?;
                    }
                }
            }
            Ok(Value::ComplexTensor(tensor))
        }
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let rows = tensor.rows();
            let storage = tensor
                .integer_storage()
                .cloned()
                .expect("integer complex branch");
            let mut real = storage.real().exact_values();
            let mut imag = storage.imag().exact_values();
            let real_zero = zero_int_like(storage.real());
            let imag_zero = zero_int_like(storage.imag());
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        let index = row + col * rows;
                        real[index] = real_zero.clone();
                        imag[index] = imag_zero.clone();
                    }
                }
            }
            Ok(Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        storage
                            .real()
                            .from_exact_values_like(real)
                            .map_err(internal)?,
                        storage
                            .imag()
                            .from_exact_values_like(imag)
                            .map_err(internal)?,
                    )
                    .map_err(internal)?,
                    shape,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        other => Err(invalid(format!(
            "decomposition: unsupported coefficient matrix {other:?}"
        ))),
    }
}

pub(super) fn symmetrize_from_triangle(
    matrix: Value,
    flag: TriangularFlag,
) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(mut tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row == col {
                        continue;
                    }
                    let value = match flag {
                        TriangularFlag::Upper if row > col => tensor.get2(col, row),
                        TriangularFlag::Lower if row < col => tensor.get2(col, row),
                        _ => continue,
                    }
                    .map_err(internal)?;
                    tensor.set2(row, col, value).map_err(internal)?;
                }
            }
            Ok(Value::Tensor(tensor))
        }
        Value::LogicalArray(mut array) => {
            let rows = logical_rows(&array);
            let cols = logical_cols(&array);
            for col in 0..cols {
                for row in 0..rows {
                    if row == col {
                        continue;
                    }
                    let source = match flag {
                        TriangularFlag::Upper if row > col => Some(col + row * rows),
                        TriangularFlag::Lower if row < col => Some(col + row * rows),
                        _ => None,
                    };
                    if let Some(source) = source {
                        array.data[row + col * rows] = array.data[source];
                    }
                }
            }
            Ok(Value::LogicalArray(array))
        }
        Value::ComplexTensor(mut tensor) if tensor.integer_storage().is_none() => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row == col {
                        continue;
                    }
                    let maybe_value = match flag {
                        TriangularFlag::Upper if row > col => tensor
                            .numeric_value_at(col + row * tensor.rows())
                            .map(|(real, imag)| (real.materialize_f64(), -imag.materialize_f64())),
                        TriangularFlag::Lower if row < col => tensor
                            .numeric_value_at(col + row * tensor.rows())
                            .map(|(real, imag)| (real.materialize_f64(), -imag.materialize_f64())),
                        _ => None,
                    };
                    if let Some(value) = maybe_value {
                        tensor
                            .set_f64_assignment_at(row + col * tensor.rows(), value.0, value.1)
                            .map_err(internal)?;
                    }
                }
            }
            Ok(Value::ComplexTensor(tensor))
        }
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let rows = tensor.rows();
            let storage = tensor
                .integer_storage()
                .cloned()
                .expect("integer complex branch");
            let mut real = storage.real().exact_values();
            let mut imag = storage.imag().exact_values();
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row == col {
                        continue;
                    }
                    let maybe_value = match flag {
                        TriangularFlag::Upper if row > col => {
                            let source = col + row * rows;
                            Some((real[source].clone(), negate_int_saturating(&imag[source])))
                        }
                        TriangularFlag::Lower if row < col => {
                            let source = col + row * rows;
                            Some((real[source].clone(), negate_int_saturating(&imag[source])))
                        }
                        _ => None,
                    };
                    if let Some(value) = maybe_value {
                        let target = row + col * rows;
                        real[target] = value.0;
                        imag[target] = value.1;
                    }
                }
            }
            Ok(Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        storage
                            .real()
                            .from_exact_values_like(real)
                            .map_err(internal)?,
                        storage
                            .imag()
                            .from_exact_values_like(imag)
                            .map_err(internal)?,
                    )
                    .map_err(internal)?,
                    shape,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        other => Err(invalid(format!(
            "decomposition: unsupported coefficient matrix {other:?}"
        ))),
    }
}
