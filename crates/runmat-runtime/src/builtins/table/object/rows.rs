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
}
