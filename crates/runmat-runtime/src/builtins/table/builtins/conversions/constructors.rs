use super::*;
use runmat_builtins::ARRAY2TABLE_GPU_INPUT_EXTENSION;
use runmat_macros::runtime_builtin;
const BUILTIN_NAME: &str = "array2table";

#[runtime_builtin(
    name = "array2table",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ARRAY2TABLE_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let mut options = parse_array2table_options(&rest)?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    validate_array2table_names(
        &names,
        &mut options.row_names,
        options.dimension_names.as_deref(),
    )?;
    let table = table_from_columns_with_properties(names, columns, options.row_names)?;
    apply_array2table_dimension_names(table, options.dimension_names)
}

fn apply_array2table_dimension_names(
    mut table: Value,
    dimension_names: Option<Vec<String>>,
) -> BuiltinResult<Value> {
    let Some(dimension_names) = dimension_names else {
        return Ok(table);
    };
    let Value::Object(object) = &mut table else {
        return Err(invalid_variable(
            "array2table: internal table construction failed",
        ));
    };
    set_table_dimension_names(object, dimension_names, "array2table")?;
    Ok(table)
}

#[runtime_builtin(
    name = "cell2table",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "cell2table")?;
    let Value::Cell(cell) = value else {
        return Err(invalid_argument("cell2table: expected cell array input"));
    };
    if cell.shape.len() > 2 {
        return Err(invalid_argument(
            "cell2table: C must be a two-dimensional cell array",
        ));
    }
    let mut columns = Vec::with_capacity(cell.cols);
    for col in 0..cell.cols {
        let mut data = Vec::with_capacity(cell.rows);
        for row in 0..cell.rows {
            data.push(cell.get(row, col).map_err(invalid_index)?);
        }
        if let Some(integer_column) = compatible_integer_column(&data, cell.rows)? {
            columns.push(integer_column);
        } else {
            columns
                .push(Value::Cell(CellArray::new(data, cell.rows, 1).map_err(
                    |err| invalid_variable(format!("cell2table: {err}")),
                )?));
        }
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "struct2table",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_struct2table_options(&rest)?;
    match value {
        Value::Struct(st) => {
            let mut names = Vec::with_capacity(st.fields.len());
            let mut columns = Vec::with_capacity(st.fields.len());
            for (name, value) in st.fields {
                names.push(name);
                if options.as_array && value_row_count(&value)? != 1 {
                    columns.push(Value::Cell(
                        CellArray::new(vec![value], 1, 1).map_err(invalid_variable)?,
                    ));
                } else {
                    columns.push(value);
                }
            }
            let names = options.table.variable_names.unwrap_or(names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        Value::StructArray(array) => {
            let rows = array.len();
            let field_names = array.field_names().cloned().collect::<Vec<_>>();
            let mut columns = Vec::with_capacity(field_names.len());
            for name in &field_names {
                let values = array
                    .field_values(name)
                    .ok_or_else(|| invalid_variable("structure array field is missing"))?
                    .to_vec();
                if let Some(integer_column) = compatible_integer_column(&values, rows)? {
                    columns.push(integer_column);
                } else {
                    columns.push(Value::Cell(
                        CellArray::new(values, rows, 1).map_err(invalid_variable)?,
                    ));
                }
            }
            let names = options.table.variable_names.unwrap_or(field_names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        other => Err(invalid_argument(format!(
            "struct2table: expected struct or struct array, got {other:?}"
        ))),
    }
}
