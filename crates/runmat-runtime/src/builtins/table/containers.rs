use super::*;

mod categorical;
mod dictionary;
mod timetable;

pub(super) use categorical::*;
pub(super) use dictionary::*;
pub(super) use timetable::*;

pub(super) async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

#[derive(Default)]
pub(super) struct TableConstructorOptions {
    pub(super) variable_names: Option<Vec<String>>,
    pub(super) row_names: Option<Vec<String>>,
}

pub(super) struct Struct2TableOptions {
    pub(super) table: TableConstructorOptions,
    pub(super) as_array: bool,
}

pub(super) fn split_table_constructor_args(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, TableConstructorOptions)> {
    let mut variables = Vec::new();
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if let Ok(name) = scalar_text(&args[idx], "table option") {
            if idx + 1 < args.len() && is_table_constructor_option(&name) {
                let value = &args[idx + 1];
                if name.eq_ignore_ascii_case("VariableNames") {
                    options.variable_names = Some(variable_name_list(value)?);
                } else if name.eq_ignore_ascii_case("RowNames") {
                    options.row_names = Some(string_list(value)?);
                }
                idx += 2;
                continue;
            }
        }
        variables.push(args[idx].clone());
        idx += 1;
    }
    Ok((variables, options))
}

pub(super) fn is_table_constructor_option(name: &str) -> bool {
    name.eq_ignore_ascii_case("VariableNames") || name.eq_ignore_ascii_case("RowNames")
}

pub(super) fn parse_table_options(
    args: &[Value],
    context: &str,
) -> BuiltinResult<TableConstructorOptions> {
    let mut options = TableConstructorOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let name = scalar_text(&args[idx], "table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            options.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            options.row_names = Some(string_list(&args[idx + 1])?);
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(options)
}

pub(super) fn parse_struct2table_options(args: &[Value]) -> BuiltinResult<Struct2TableOptions> {
    let mut table = TableConstructorOptions::default();
    let mut as_array = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "struct2table: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "struct2table option")?;
        if name.eq_ignore_ascii_case("VariableNames") {
            table.variable_names = Some(variable_name_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("RowNames") {
            table.row_names = Some(string_list(&args[idx + 1])?);
        } else if name.eq_ignore_ascii_case("AsArray") {
            as_array = bool_scalar(&args[idx + 1], "AsArray")?;
        } else {
            return Err(invalid_argument(format!(
                "struct2table: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(Struct2TableOptions { table, as_array })
}

pub(super) fn parse_table2struct_to_scalar(args: &[Value]) -> BuiltinResult<bool> {
    let mut to_scalar = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "table2struct: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "table2struct option")?;
        if name.eq_ignore_ascii_case("ToScalar") {
            to_scalar = bool_scalar(&args[idx + 1], "ToScalar")?;
        } else {
            return Err(invalid_argument(format!(
                "table2struct: unsupported option '{name}'"
            )));
        }
        idx += 2;
    }
    Ok(to_scalar)
}

pub(super) fn split_readtimetable_options(
    args: &[Value],
) -> BuiltinResult<(Vec<Value>, Vec<Value>)> {
    let mut readtable_args = Vec::new();
    let mut timetable_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(
                "readtimetable: name-value options must be provided in pairs",
            ));
        }
        let name = scalar_text(&args[idx], "readtimetable option")?;
        if name.eq_ignore_ascii_case(ROW_TIMES) {
            timetable_args.push(args[idx].clone());
            timetable_args.push(args[idx + 1].clone());
        } else {
            readtable_args.push(args[idx].clone());
            readtable_args.push(args[idx + 1].clone());
        }
        idx += 2;
    }
    Ok((readtable_args, timetable_args))
}

pub(super) fn parse_named_option<'a>(args: &'a [Value], name: &str) -> Option<&'a Value> {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if scalar_text(&args[idx], "option name")
            .map(|text| text.eq_ignore_ascii_case(name))
            .unwrap_or(false)
        {
            return args.get(idx + 1);
        }
        idx += 2;
    }
    None
}

pub(super) fn parse_bool_option(
    args: &[Value],
    name: &str,
    default: bool,
    context: &str,
) -> BuiltinResult<bool> {
    let mut result = default;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = bool_scalar(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}

pub(super) fn parse_named_text_option(
    args: &[Value],
    name: &str,
    default: &str,
    context: &str,
) -> BuiltinResult<String> {
    let mut result = default.to_string();
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid_argument(format!(
                "{context}: name-value options must be provided in pairs"
            )));
        }
        let option_name = scalar_text(&args[idx], "option name")?;
        if option_name.eq_ignore_ascii_case(name) {
            result = scalar_text(&args[idx + 1], name)?;
        } else {
            return Err(invalid_argument(format!(
                "{context}: unsupported option '{option_name}'"
            )));
        }
        idx += 2;
    }
    Ok(result)
}

pub(super) fn split_value_columns(value: Value) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut out = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(tensor.get2(row, col).map_err(invalid_index)?);
                }
                out.push(Value::Tensor(
                    Tensor::new_with_dtype(data, vec![rows, 1], tensor.dtype)
                        .map_err(invalid_variable)?,
                ));
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

pub(super) fn table_to_cell_array(object: &ObjectInstance) -> BuiltinResult<Value> {
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

pub(super) fn row_value(value: &Value, row: usize) -> BuiltinResult<Value> {
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

pub(super) fn colon_colon_payload() -> Value {
    Value::Cell(
        CellArray::new(vec![Value::from(":"), Value::from(":")], 1, 2)
            .expect("selector cell shape is valid"),
    )
}
