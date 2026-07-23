use super::*;

pub(crate) fn selected_row_names(
    object: &ObjectInstance,
    rows: &[usize],
) -> BuiltinResult<Option<Vec<String>>> {
    let props = table_public_properties(object)?;
    let Some(value) = props.fields.get(ROW_NAMES) else {
        return Ok(None);
    };
    let names = string_list(value)?;
    if names.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        rows.iter()
            .filter_map(|row| names.get(*row).cloned())
            .collect(),
    ))
}

pub(super) fn selected_row_times(
    object: &ObjectInstance,
    rows: &[usize],
) -> BuiltinResult<Option<Value>> {
    let Some(row_times) = timetable_row_times(object)? else {
        return Ok(None);
    };
    select_rows(&row_times, rows).map(Some)
}

pub(crate) fn value_row_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) => Ok(tensor.rows()),
        Value::ComplexTensor(tensor) => Ok(tensor.rows),
        Value::StringArray(array) => Ok(array.rows()),
        Value::LogicalArray(array) => Ok(array.shape.first().copied().unwrap_or(array.data.len())),
        Value::Cell(cell) => Ok(cell.rows),
        Value::CharArray(array) => Ok(array.rows),
        Value::Object(obj) if obj.is_class("datetime") => {
            crate::builtins::datetime::serials_from_datetime_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class("duration") => {
            crate::builtins::duration::duration_tensor_from_duration_value(value)
                .map(|tensor| tensor.rows())
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => obj
            .properties
            .get("Codes")
            .map(value_row_count)
            .transpose()
            .map(|rows| rows.unwrap_or(0)),
        Value::Object(obj) if is_tabular_class(obj) => table_height(obj),
        _ => Ok(1),
    }
}

pub(crate) fn select_rows(value: &Value, rows: &[usize]) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let cols = tensor.cols();
            if let Some(storage) = tensor.integer_storage() {
                let mut values = Vec::with_capacity(rows.len() * cols);
                for col in 0..cols {
                    for &row in rows {
                        values.push(storage.value_at(row + col * tensor.rows()).ok_or_else(
                            || invalid_index("table: numeric variable row index out of bounds"),
                        )?);
                    }
                }
                return Tensor::new_integer(
                    storage
                        .from_exact_values_like(values)
                        .map_err(invalid_variable)?,
                    vec![rows.len(), cols],
                )
                .map(Value::Tensor)
                .map_err(invalid_variable);
            }
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
            }
            Tensor::new_with_dtype(data, vec![rows.len(), cols], tensor.dtype)
                .map(Value::Tensor)
                .map_err(invalid_variable)
        }
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = &tensor.integer_data {
                let mut real = Vec::with_capacity(rows.len() * tensor.cols);
                let mut imag = Vec::with_capacity(rows.len() * tensor.cols);
                for col in 0..tensor.cols {
                    for &row in rows {
                        let index = row + col * tensor.rows;
                        real.push(storage.real.value_at(index).ok_or_else(|| {
                            invalid_index("table: complex variable row index out of bounds")
                        })?);
                        imag.push(storage.imag.value_at(index).ok_or_else(|| {
                            invalid_index("table: complex variable row index out of bounds")
                        })?);
                    }
                }
                return ComplexTensor::new_integer(
                    runmat_builtins::IntegerComplexStorage::new(
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
                    vec![rows.len(), tensor.cols],
                )
                .map(Value::ComplexTensor)
                .map_err(invalid_variable);
            }
            let mut data = Vec::with_capacity(rows.len() * tensor.cols);
            for col in 0..tensor.cols {
                for &row in rows {
                    let idx = row + col * tensor.rows;
                    data.push(*tensor.data.get(idx).ok_or_else(|| {
                        invalid_index("table: complex variable row index out of bounds")
                    })?);
                }
            }
            ComplexTensor::new(data, vec![rows.len(), tensor.cols])
                .map(Value::ComplexTensor)
                .map_err(invalid_variable)
        }
        Value::StringArray(array) => {
            let cols = array.cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * array.rows();
                    data.push(array.data.get(idx).cloned().ok_or_else(|| {
                        invalid_index("table: string variable row index out of bounds")
                    })?);
                }
            }
            StringArray::new(data, vec![rows.len(), cols])
                .map(Value::StringArray)
                .map_err(invalid_variable)
        }
        Value::CharArray(array) => {
            let mut data = Vec::with_capacity(rows.len() * array.cols);
            for &row in rows {
                if row >= array.rows {
                    return Err(invalid_index(
                        "table: char variable row index out of bounds",
                    ));
                }
                let start = row * array.cols;
                data.extend_from_slice(&array.data[start..start + array.cols]);
            }
            CharArray::new(data, rows.len(), array.cols)
                .map(Value::CharArray)
                .map_err(invalid_variable)
        }
        Value::LogicalArray(array) => {
            let source_rows = array.shape.first().copied().unwrap_or(array.data.len());
            let cols = array.shape.get(1).copied().unwrap_or(1);
            let mut data = Vec::with_capacity(rows.len() * cols);
            for col in 0..cols {
                for &row in rows {
                    let idx = row + col * source_rows;
                    data.push(*array.data.get(idx).ok_or_else(|| {
                        invalid_index("table: logical variable row index out of bounds")
                    })?);
                }
            }
            LogicalArray::new(data, vec![rows.len(), cols])
                .map(Value::LogicalArray)
                .map_err(invalid_variable)
        }
        Value::Cell(cell) => {
            let mut data = Vec::with_capacity(rows.len() * cell.cols);
            for col in 0..cell.cols {
                for &row in rows {
                    data.push(cell.get(row, col).map_err(invalid_index)?);
                }
            }
            CellArray::new(data, rows.len(), cell.cols)
                .map(Value::Cell)
                .map_err(invalid_variable)
        }
        Value::Object(obj) if obj.is_class("datetime") => {
            let tensor = crate::builtins::datetime::serials_from_datetime_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::datetime::datetime_object_from_serial_tensor(
                        tensor,
                        crate::builtins::datetime::datetime_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        Value::Object(obj) if obj.is_class("duration") => {
            let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
            let selected = select_rows(&Value::Tensor(tensor), rows)?;
            match selected {
                Value::Tensor(tensor) => {
                    crate::builtins::duration::duration_object_from_days_tensor(
                        tensor,
                        crate::builtins::duration::duration_format_from_value(value),
                    )
                }
                _ => unreachable!("select_rows tensor branch returns tensor"),
            }
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => {
            let codes = obj
                .properties
                .get("Codes")
                .ok_or_else(|| invalid_variable("categorical: missing Codes property"))?;
            let selected_codes = select_rows(codes, rows)?;
            let mut out = obj.clone();
            out.properties.insert("Codes".to_string(), selected_codes);
            Ok(Value::Object(out))
        }
        _ if rows.len() == 1 && rows[0] == 0 => Ok(value.clone()),
        other => Err(invalid_variable(format!(
            "table: row selection unsupported for variable {other:?}"
        ))),
    }
}

pub(super) fn assign_rows(mut current: Value, rows: &[usize], rhs: Value) -> BuiltinResult<Value> {
    if value_row_count(&rhs)? != rows.len() {
        return Err(invalid_variable(
            "table: assignment row count must match selected row count",
        ));
    }
    let replacing_all_rows = rows.len() == value_row_count(&current)?;
    match (&mut current, rhs) {
        (Value::Tensor(target), Value::Tensor(source)) => {
            if target.cols() != source.cols() {
                return Err(invalid_variable(
                    "table: tensor assignment column count mismatch",
                ));
            }
            let target_rows = target.rows();
            let target_cols = target.cols();
            if target.integer_storage().is_some() {
                let target_storage = target
                    .integer_data
                    .as_mut()
                    .expect("integer storage was checked above");
                for col in 0..target_cols {
                    for (src_row, &dst_row) in rows.iter().enumerate() {
                        let source_index = src_row + col * source.rows();
                        let target_index = dst_row + col * target_rows;
                        let exact = match source.integer_storage() {
                            Some(source_storage) => target_storage.cast_exact_assignment(
                                &source_storage.value_at(source_index).ok_or_else(|| {
                                    invalid_index("table: source integer storage is inconsistent")
                                })?,
                            ),
                            None => target_storage.cast_f64_assignment(
                                source.get2(src_row, col).map_err(invalid_index)?,
                            ),
                        };
                        let compatibility_value = exact.to_f64();
                        target_storage
                            .set_value(target_index, exact)
                            .map_err(invalid_variable)?;
                        target.data[target_index] = compatibility_value;
                    }
                }
                return Ok(current);
            }
            for col in 0..target.cols() {
                for (src_row, &dst_row) in rows.iter().enumerate() {
                    let value = source.get2(src_row, col).map_err(invalid_index)?;
                    target.set2(dst_row, col, value).map_err(invalid_index)?;
                }
            }
            Ok(current)
        }
        (_, source) if replacing_all_rows => Ok(source),
        _ => Err(invalid_variable(
            "table: assignment for this variable type requires replacing all rows",
        )),
    }
}

pub(super) fn concatenate_numeric_columns(values: &[&Value]) -> BuiltinResult<Value> {
    let rows = values
        .first()
        .and_then(|value| match value {
            Value::Tensor(t) => Some(t.rows()),
            _ => None,
        })
        .unwrap_or(0);
    let cols = values
        .iter()
        .map(|value| match value {
            Value::Tensor(t) => Ok(t.cols()),
            _ => Err(invalid_variable("table: expected numeric variable")),
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    let total_cols: usize = cols.iter().sum();
    let typed_prototype = values.iter().find_map(|value| match value {
        Value::Tensor(tensor) => tensor.integer_storage(),
        _ => None,
    });
    if let Some(prototype) = typed_prototype {
        let all_same_typed_class = values.iter().all(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some_and(|storage| storage.class_name() == prototype.class_name()))
        });
        if all_same_typed_class {
            let mut exact = Vec::with_capacity(rows * total_cols);
            for value in values {
                let Value::Tensor(tensor) = value else {
                    unreachable!("numeric columns were validated above");
                };
                let storage = tensor
                    .integer_storage()
                    .expect("typed column was checked above");
                for col in 0..tensor.cols() {
                    for row in 0..rows {
                        exact.push(storage.value_at(row + col * tensor.rows()).ok_or_else(
                            || invalid_index("table: integer column row index out of bounds"),
                        )?);
                    }
                }
            }
            return Tensor::new_integer(
                prototype
                    .from_same_class_values(exact)
                    .map_err(invalid_variable)?,
                vec![rows, total_cols],
            )
            .map(Value::Tensor)
            .map_err(invalid_variable);
        }
    }
    let mut data = Vec::with_capacity(rows * total_cols);
    for value in values {
        let Value::Tensor(tensor) = value else {
            return Err(invalid_variable("table: expected numeric variable"));
        };
        for col in 0..tensor.cols() {
            for row in 0..rows {
                data.push(tensor.get2(row, col).map_err(invalid_index)?);
            }
        }
    }
    Tensor::new(data, vec![rows, total_cols])
        .map(Value::Tensor)
        .map_err(invalid_variable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::IntegerStorage;

    #[test]
    fn select_rows_preserves_exact_integer_storage() {
        let large = 9_007_199_254_740_993_u64;
        let value = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX, 7, 0]), vec![2, 2])
                .unwrap(),
        );

        let selected = select_rows(&value, &[1, 0]).unwrap();
        let Value::Tensor(selected) = selected else {
            panic!("expected tensor row selection");
        };
        assert_eq!(
            selected.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, large, 0, 7]))
        );
    }

    #[test]
    fn select_rows_preserves_exact_complex_integer_storage() {
        let large = 9_007_199_254_740_993_i64;
        let value = Value::ComplexTensor(
            ComplexTensor::new_integer(
                runmat_builtins::IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![large, i64::MIN]),
                    IntegerStorage::I64(vec![0, 7]),
                )
                .unwrap(),
                vec![2, 1],
            )
            .unwrap(),
        );

        let selected = select_rows(&value, &[1, 0]).unwrap();
        let Value::ComplexTensor(selected) = selected else {
            panic!("expected complex tensor row selection");
        };
        assert_eq!(
            selected.integer_data,
            Some(
                runmat_builtins::IntegerComplexStorage::new(
                    IntegerStorage::I64(vec![i64::MIN, large]),
                    IntegerStorage::I64(vec![7, 0]),
                )
                .unwrap(),
            )
        );
    }

    #[test]
    fn assign_rows_preserves_exact_integer_source_and_target_storage() {
        let large = 9_007_199_254_740_993_u64;
        let current = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![0, 1]), vec![2, 1]).unwrap(),
        );
        let rhs = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap(),
        );

        let Value::Tensor(result) = assign_rows(current, &[1], rhs).unwrap() else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![0, large]))
        );
    }

    #[test]
    fn assign_rows_converts_floating_source_into_target_integer_class() {
        let current =
            Value::Tensor(Tensor::new_integer(IntegerStorage::I8(vec![0, 0]), vec![2, 1]).unwrap());
        let rhs = Value::Tensor(Tensor::new_2d(vec![200.6], 1, 1).unwrap());

        let Value::Tensor(result) = assign_rows(current, &[0], rhs).unwrap() else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I8(vec![i8::MAX, 0]))
        );
    }

    #[test]
    fn concatenate_numeric_columns_preserves_same_class_exact_integers() {
        let large = 9_007_199_254_740_993_u64;
        let first = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![large, u64::MAX]), vec![2, 1]).unwrap(),
        );
        let second = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![7, 0, 9, 2]), vec![2, 2]).unwrap(),
        );

        let Value::Tensor(result) = concatenate_numeric_columns(&[&first, &second]).unwrap() else {
            panic!("expected tensor result");
        };
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![large, u64::MAX, 7, 0, 9, 2]))
        );
    }
}
