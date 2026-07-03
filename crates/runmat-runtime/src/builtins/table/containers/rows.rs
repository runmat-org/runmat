use super::*;

pub(in crate::builtins::table) fn table_to_cell_array(
    object: &ObjectInstance,
) -> BuiltinResult<Value> {
    let height = table_height(object)?;
    let names = table_variable_names_from_object(object)?;
    let variables = table_variables(object)?;
    let mut data = Vec::with_capacity(height * names.len());
    for col_name in &names {
        let value = variables.fields.get(col_name).ok_or_else(|| {
            invalid_variable(format!("table2cell: missing variable '{col_name}'"))
        })?;
        for row in 0..height {
            data.push(row_value(value, row)?);
        }
    }
    CellArray::new(data, height, names.len())
        .map(Value::Cell)
        .map_err(invalid_variable)
}

pub(in crate::builtins::table) fn row_value(value: &Value, row: usize) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) if tensor.cols() == 1 => {
            Ok(Value::Num(tensor.get2(row, 0).map_err(invalid_index)?))
        }
        Value::Tensor(_) => select_rows(value, &[row]),
        Value::ComplexTensor(tensor) if tensor.cols == 1 => {
            let value = tensor
                .data
                .get(row)
                .copied()
                .ok_or_else(|| invalid_index("table2cell: complex row out of bounds"))?;
            Ok(Value::Complex(value.0, value.1))
        }
        Value::ComplexTensor(_) => select_rows(value, &[row]),
        Value::StringArray(array) if array.cols() == 1 => Ok(Value::String(
            array.data.get(row).cloned().unwrap_or_default(),
        )),
        Value::StringArray(_) => select_rows(value, &[row]),
        Value::LogicalArray(array) if array.shape.get(1).copied().unwrap_or(1) == 1 => {
            Ok(Value::Bool(*array.data.get(row).unwrap_or(&0) != 0))
        }
        Value::LogicalArray(_) => select_rows(value, &[row]),
        Value::Cell(cell) if cell.cols == 1 => cell.get(row, 0).map_err(invalid_index),
        Value::Cell(_) => select_rows(value, &[row]),
        Value::CharArray(array) => {
            if row >= array.rows {
                return Err(invalid_index("table2cell: char row out of bounds"));
            }
            let start = row * array.cols;
            CharArray::new(
                array.data[start..start + array.cols].to_vec(),
                1,
                array.cols,
            )
            .map(Value::CharArray)
            .map_err(invalid_variable)
        }
        Value::Object(obj) if obj.is_class("datetime") || obj.is_class("duration") => {
            select_rows(value, &[row])
        }
        Value::Object(obj) if obj.is_class(CATEGORICAL_CLASS) => select_rows(value, &[row]),
        other if row == 0 => Ok(other.clone()),
        _ => Err(invalid_index("table2cell: row out of bounds")),
    }
}

pub(in crate::builtins::table) fn colon_colon_payload() -> Value {
    Value::Cell(
        CellArray::new(vec![Value::from(":"), Value::from(":")], 1, 2)
            .expect("selector cell shape is valid"),
    )
}
