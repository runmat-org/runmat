//! MATLAB table, timetable, categorical, and tabular workflow builtins.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use calamine::{open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use encoding_rs::{Encoding, UTF_8};
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericDType, ObjectInstance, StringArray,
    StructValue, Tensor, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::{
    gather_if_needed_async, BuiltinResult, OBJECT_INDEX_BRACE, OBJECT_INDEX_MEMBER,
    OBJECT_INDEX_PAREN,
};

mod containers;
mod display;
mod import;
mod metadata;
mod object;
mod registry;

pub use metadata::*;
pub use registry::ensure_table_class_registered;

use containers::*;
use display::{categorical_label_at, format_key_number};
pub use display::{table_display_text, table_summary_text};
use import::*;
use object::*;
pub use object::{
    is_table_value, is_tabular_object, sortrows_table, table_from_columns, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
pub(crate) use object::{
    select_rows, selected_row_names, table_from_columns_like, value_row_count,
};
pub const TABLE_CLASS: &str = "table";
pub const TIMETABLE_CLASS: &str = "timetable";
const CATEGORICAL_CLASS: &str = "categorical";
const DICTIONARY_CLASS: &str = "dictionary";
const TIMERANGE_CLASS: &str = "timerange";
const VARTYPE_CLASS: &str = "vartype";
const ROWFILTER_CLASS: &str = "rowfilter";
const ARRAY_DATASTORE_CLASS: &str = "arrayDatastore";
const PARQUET_DATASTORE_CLASS: &str = "parquetDatastore";
const UITABLE_CLASS: &str = "uitable";
const TABLE_VARIABLES_FIELD: &str = "__table_variables";
const TABLE_PROPERTIES_FIELD: &str = "__table_properties";
const PROPERTIES_MEMBER: &str = "Properties";
const ROW_TIMES: &str = "RowTimes";
const VARIABLE_NAMES: &str = "VariableNames";
const ROW_NAMES: &str = "RowNames";
const DIMENSION_NAMES: &str = "DimensionNames";
const VARIABLE_UNITS: &str = "VariableUnits";
const VARIABLE_DESCRIPTIONS: &str = "VariableDescriptions";
const DESCRIPTION: &str = "Description";
const USER_DATA: &str = "UserData";
const DEFAULT_ROW_DIM_NAME: &str = "Rows";
const DEFAULT_VARIABLE_DIM_NAME: &str = "Variables";

#[runtime_builtin(
    name = "table",
    category = "table",
    summary = "Create a table from named column variables.",
    keywords = "table,VariableNames,RowNames,Properties",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::TABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let gathered = gather_values(&args).await?;
    let (variables, options) = split_table_constructor_args(gathered)?;
    let names = if let Some(names) = options.variable_names {
        names
    } else {
        generated_variable_names(variables.len())
    };
    table_from_columns_with_properties(names, variables, options.row_names)
}

#[runtime_builtin(
    name = "readtable",
    category = "io/tabular",
    summary = "Import tabular text or spreadsheet data into a table.",
    keywords = "readtable,table,csv,tsv,xlsx,xls,ods,spreadsheet,VariableNames,RowNames,Sheet,Range",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::READTABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readtable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    read_table_from_file(&resolved, &options).await
}

#[runtime_builtin(
    name = "spreadsheetImportOptions",
    category = "io/tabular",
    summary = "Create spreadsheet import options for readtable.",
    keywords = "spreadsheetImportOptions,readtable,spreadsheet,xlsx,xls,DataRange,VariableTypes,VariableNames,NumVariables",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn spreadsheet_import_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_values(&args).await?;
    spreadsheet_import_options(gathered)
}

#[runtime_builtin(
    name = "detectImportOptions",
    category = "io/tabular",
    summary = "Inspect a text or spreadsheet file and create import options.",
    keywords = "detectImportOptions,readtable,readmatrix,csv,tsv,xlsx,Delimiter,VariableTypes,VariableNames",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::DETECT_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn detect_import_options_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    detect_import_options_from_file(&resolved, &options).await
}

#[runtime_builtin(
    name = "height",
    category = "table",
    summary = "Return the number of rows in a table.",
    keywords = "height,table,rows",
    descriptor(crate::builtins::table::HEIGHT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn height_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_height(object)? as f64));
    }
    value_row_count(&host).map(|n| Value::Num(n as f64))
}

#[runtime_builtin(
    name = "width",
    category = "table",
    summary = "Return the number of variables in a table.",
    keywords = "width,table,variables",
    descriptor(crate::builtins::table::WIDTH_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn width_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    if let Some(object) = table_object(&host) {
        return Ok(Value::Num(table_width(object)? as f64));
    }
    match host {
        Value::Tensor(t) => Ok(Value::Num(t.cols() as f64)),
        Value::ComplexTensor(t) => Ok(Value::Num(t.cols as f64)),
        Value::StringArray(sa) => Ok(Value::Num(sa.cols() as f64)),
        Value::LogicalArray(la) => Ok(Value::Num(la.shape.get(1).copied().unwrap_or(1) as f64)),
        Value::Cell(ca) => Ok(Value::Num(ca.cols as f64)),
        Value::CharArray(ca) => Ok(Value::Num(ca.cols as f64)),
        _ => Ok(Value::Num(1.0)),
    }
}

#[runtime_builtin(
    name = "istable",
    category = "table",
    summary = "Return true for table arrays.",
    keywords = "istable,table,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn istable_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(TABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "istimetable",
    category = "table",
    summary = "Return true for timetable arrays.",
    keywords = "istimetable,timetable,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn istimetable_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(TIMETABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "iscategorical",
    category = "table",
    summary = "Return true for categorical arrays.",
    keywords = "iscategorical,categorical,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn iscategorical_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(CATEGORICAL_CLASS)
    )))
}

#[runtime_builtin(
    name = "array2table",
    category = "table",
    summary = "Convert an array into a table.",
    keywords = "array2table,table,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "array2table")?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "cell2table",
    category = "table",
    summary = "Convert a cell array into a table.",
    keywords = "cell2table,table,cell,VariableNames,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn cell2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_table_options(&rest, "cell2table")?;
    let Value::Cell(cell) = value else {
        return Err(invalid_argument("cell2table: expected cell array input"));
    };
    let mut columns = Vec::with_capacity(cell.cols);
    for col in 0..cell.cols {
        let mut data = Vec::with_capacity(cell.rows);
        for row in 0..cell.rows {
            data.push(cell.get(row, col).map_err(invalid_index)?);
        }
        columns
            .push(Value::Cell(CellArray::new(data, cell.rows, 1).map_err(
                |err| invalid_variable(format!("cell2table: {err}")),
            )?));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    table_from_columns_with_properties(names, columns, options.row_names)
}

#[runtime_builtin(
    name = "struct2table",
    category = "table",
    summary = "Convert a scalar struct into a table.",
    keywords = "struct2table,table,struct,AsArray,RowNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn struct2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
        Value::Cell(cell)
            if cell
                .data
                .iter()
                .all(|value| matches!(value, Value::Struct(_))) =>
        {
            let rows = cell.data.len();
            let first = cell.data.iter().find_map(|value| match value {
                Value::Struct(st) => Some(st),
                _ => None,
            });
            let field_names = first
                .map(|st| st.fields.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let mut columns = Vec::with_capacity(field_names.len());
            for name in &field_names {
                let mut values = Vec::with_capacity(rows);
                for value in &cell.data {
                    let Value::Struct(st) = value else {
                        unreachable!("checked above")
                    };
                    values.push(st.fields.get(name).cloned().unwrap_or(Value::Num(f64::NAN)));
                }
                columns.push(Value::Cell(
                    CellArray::new(values, rows, 1).map_err(invalid_variable)?,
                ));
            }
            let names = options.table.variable_names.unwrap_or(field_names);
            table_from_columns_with_properties(names, columns, options.table.row_names)
        }
        other => Err(invalid_argument(format!(
            "struct2table: expected struct or struct array, got {other:?}"
        ))),
    }
}

#[runtime_builtin(
    name = "table2struct",
    category = "table",
    summary = "Convert a table into row structs or a scalar struct of variables.",
    keywords = "table2struct,table,struct,ToScalar",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let to_scalar = parse_table2struct_to_scalar(&rest)?;
    let object = into_table_object(host, "table2struct")?;
    if to_scalar {
        return Ok(Value::Struct(table_variables(&object)?));
    }
    let height = table_height(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut rows = Vec::with_capacity(height);
    for row in 0..height {
        let mut st = StructValue::new();
        for name in &names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("table2struct: missing variable '{name}'"))
            })?;
            st.insert(name.clone(), row_value(value, row)?);
        }
        rows.push(Value::Struct(st));
    }
    CellArray::new(rows, height, 1)
        .map(Value::Cell)
        .map_err(invalid_variable)
}

#[runtime_builtin(
    name = "table2array",
    category = "table",
    summary = "Convert table variables into a homogeneous array when possible.",
    keywords = "table2array,table,array",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2array")?;
    table_brace_get(&object, &colon_colon_payload())
}

#[runtime_builtin(
    name = "table2cell",
    category = "table",
    summary = "Convert table variables into a cell array.",
    keywords = "table2cell,table,cell",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2cell")?;
    table_to_cell_array(&object)
}

#[runtime_builtin(
    name = "head",
    category = "table",
    summary = "Return the first rows of a table, timetable, or array.",
    keywords = "head,table,timetable,preview,rows",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn head_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let n = rest
        .first()
        .map(|value| nonnegative_usize(value, "head row count"))
        .transpose()?
        .unwrap_or(8);
    let rows = value_row_count(&value)?;
    let selected = (0..rows.min(n)).collect::<Vec<_>>();
    if let Some(object) = table_object(&value) {
        let names = table_variable_names_from_object(object)?;
        let variables = table_variables(object)?;
        let mut columns = Vec::with_capacity(names.len());
        for name in &names {
            columns.push(select_rows(
                variables
                    .fields
                    .get(name)
                    .ok_or_else(|| invalid_variable(format!("head: missing variable '{name}'")))?,
                &selected,
            )?);
        }
        return subset_tabular_object(object, names, columns, &selected);
    }
    select_rows(&value, &selected)
}

#[runtime_builtin(
    name = "timetable",
    category = "table",
    summary = "Create a timetable from row times and variables.",
    keywords = "timetable,table,RowTimes,TimeStep,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timetable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let (row_times, variables, options) = split_timetable_constructor_args(args)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(variables.len()));
    let mut value =
        table_from_columns_with_class(TIMETABLE_CLASS, names, variables, options.row_names)?;
    if let Value::Object(object) = &mut value {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(value)
}

#[runtime_builtin(
    name = "array2timetable",
    category = "table",
    summary = "Convert an array into a timetable.",
    keywords = "array2timetable,timetable,RowTimes,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (row_times, options) = parse_timetable_options(&rest, "array2timetable")?;
    let columns = split_value_columns(value)?;
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(columns.len()));
    let mut out =
        table_from_columns_with_class(TIMETABLE_CLASS, names, columns, options.row_names)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, row_times)?;
    }
    Ok(out)
}

#[runtime_builtin(
    name = "table2timetable",
    category = "table",
    summary = "Convert a table into a timetable.",
    keywords = "table2timetable,timetable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table2timetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (row_times, _options) = parse_timetable_options(&rest, "table2timetable")?;
    let object = into_table_object(host, "table2timetable")?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    let (times, out_names) = if let Some(row_times) = row_times {
        (Some(row_times), names)
    } else if let Some(first) = names.first() {
        let first_value = variables.fields.get(first).cloned();
        if first_value
            .as_ref()
            .map(is_time_like_value)
            .unwrap_or(false)
        {
            (first_value, names[1..].to_vec())
        } else {
            (None, names)
        }
    } else {
        (None, names)
    };
    let mut out_columns = Vec::with_capacity(out_names.len());
    for name in &out_names {
        out_columns.push(variables.fields.get(name).cloned().ok_or_else(|| {
            invalid_variable(format!("table2timetable: missing variable '{name}'"))
        })?);
    }
    let mut out =
        table_from_columns_with_class(TIMETABLE_CLASS, out_names, out_columns, row_names)?;
    if let Value::Object(object) = &mut out {
        set_timetable_row_times(object, times)?;
    }
    Ok(out)
}

#[runtime_builtin(
    name = "timetable2table",
    category = "table",
    summary = "Convert a timetable into a table.",
    keywords = "timetable2table,timetable,table,ConvertRowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timetable2table_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let convert_row_times = parse_bool_option(&rest, "ConvertRowTimes", false, "timetable2table")?;
    let object = into_timetable_object(host, "timetable2table")?;
    let mut names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut columns = Vec::with_capacity(names.len() + usize::from(convert_row_times));
    if convert_row_times {
        if let Some(row_times) = timetable_row_times(&object)? {
            columns.push(row_times);
            names.insert(0, "Time".to_string());
        }
    }
    for name in table_variable_names_from_object(&object)? {
        columns.push(variables.fields.get(&name).cloned().ok_or_else(|| {
            invalid_variable(format!("timetable2table: missing variable '{name}'"))
        })?);
    }
    let row_names = selected_row_names(&object, &(0..table_height(&object)?).collect::<Vec<_>>())?;
    table_from_columns_with_properties(names, columns, row_names)
}

#[runtime_builtin(
    name = "readtimetable",
    category = "io/tabular",
    summary = "Read tabular data into a timetable.",
    keywords = "readtimetable,timetable,readtable,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readtimetable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let (readtable_args, timetable_args) = split_readtimetable_options(&rest)?;
    let table = readtable_builtin(path, readtable_args).await?;
    table2timetable_builtin(table, timetable_args).await
}

#[runtime_builtin(
    name = "writetable",
    category = "io/tabular",
    summary = "Write a table to a delimited text file.",
    keywords = "writetable,table,csv,delimited text,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn writetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    write_tabular_file(value, rest, false).await
}

#[runtime_builtin(
    name = "writetimetable",
    category = "io/tabular",
    summary = "Write a timetable to a delimited text file.",
    keywords = "writetimetable,timetable,csv,delimited text,RowTimes",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn writetimetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    write_tabular_file(value, rest, true).await
}

#[runtime_builtin(
    name = "readcell",
    category = "io/tabular",
    summary = "Read text or spreadsheet data into a cell array.",
    keywords = "readcell,cell,readtable,csv,spreadsheet",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn readcell_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(&rest)?;
    read_cell_from_file(&path, &options).await
}

#[runtime_builtin(
    name = "categorical",
    category = "table",
    summary = "Create a categorical array.",
    keywords = "categorical,categories,ordinal,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn categorical_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    categorical_from_args(args)
}

#[runtime_builtin(
    name = "dictionary",
    category = "table",
    summary = "Create a key-value dictionary object.",
    keywords = "dictionary,containers.Map,key,value,map",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    dictionary_from_args(args)
}

#[runtime_builtin(
    name = "timerange",
    category = "table",
    summary = "Create a timetable row-time range selector.",
    keywords = "timerange,timetable,row times",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn timerange_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    if args.len() > 3 {
        return Err(invalid_argument(
            "timerange: expected start, end, and optional inclusivity",
        ));
    }
    let gathered = gather_values(&args).await?;
    let mut object = ObjectInstance::new(TIMERANGE_CLASS.to_string());
    object.properties.insert(
        "Start".to_string(),
        gathered
            .first()
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "End".to_string(),
        gathered
            .get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    object.properties.insert(
        "Inclusivity".to_string(),
        gathered
            .get(2)
            .cloned()
            .unwrap_or_else(|| Value::from("closed")),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "vartype",
    category = "table",
    summary = "Create a table variable type selector.",
    keywords = "vartype,table,selector,variable type",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn vartype_builtin(value: Value) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let mut object = ObjectInstance::new(VARTYPE_CLASS.to_string());
    object.properties.insert("Type".to_string(), value);
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "rowfilter",
    category = "table",
    summary = "Create a table row filter descriptor.",
    keywords = "rowfilter,table,rows,filter",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn rowfilter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(ROWFILTER_CLASS.to_string());
    object.properties.insert(
        "Variables".to_string(),
        args.first()
            .cloned()
            .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
    );
    object.properties.insert(
        "Predicate".to_string(),
        args.get(1)
            .cloned()
            .unwrap_or_else(|| Value::String(String::new())),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "pivot",
    category = "table",
    summary = "Pivot or summarize table data by grouping variables.",
    keywords = "pivot,table,reshape,groupsummary",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn pivot_builtin(
    table: Value,
    rowvars: Value,
    colvars: Value,
    datavar: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let rowvars = gather_if_needed_async(&rowvars)
        .await
        .map_err(map_control_flow)?;
    let colvars = gather_if_needed_async(&colvars)
        .await
        .map_err(map_control_flow)?;
    let datavar = gather_if_needed_async(&datavar)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let method = parse_named_text_option(&rest, "Method", "sum", "pivot")?;
    pivot_impl(table, rowvars, colvars, datavar, &method)
}

#[runtime_builtin(
    name = "arrayDatastore",
    category = "io/tabular",
    summary = "Create an array datastore descriptor.",
    keywords = "arrayDatastore,datastore,array,data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn array_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(ARRAY_DATASTORE_CLASS.to_string());
    object.properties.insert(
        "Data".to_string(),
        args.first()
            .cloned()
            .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap())),
    );
    object
        .properties
        .insert("ReadSize".to_string(), Value::Num(1.0));
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "parquetDatastore",
    category = "io/tabular",
    summary = "Create a parquet datastore descriptor.",
    keywords = "parquetDatastore,datastore,parquet,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn parquet_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(PARQUET_DATASTORE_CLASS.to_string());
    object.properties.insert(
        "Files".to_string(),
        args.first().cloned().unwrap_or_else(|| {
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap())
        }),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "uitable",
    category = "table",
    summary = "Create a table UI component descriptor.",
    keywords = "uitable,ui,table,Data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn uitable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    let mut object = ObjectInstance::new(UITABLE_CLASS.to_string());
    let data = parse_named_option(&args, "Data")
        .cloned()
        .or_else(|| args.first().cloned())
        .unwrap_or_else(|| Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap()));
    object.properties.insert("Data".to_string(), data);
    object.properties.insert(
        "ColumnName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    object.properties.insert(
        "RowName".to_string(),
        Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
    );
    Ok(Value::Object(object))
}

#[runtime_builtin(
    name = "groupsummary",
    category = "table",
    summary = "Group table rows and compute summary statistics for data variables.",
    keywords = "groupsummary,group,table,mean,sum,count,median,min,max",
    accel = "cpu",
    descriptor(crate::builtins::table::GROUPSUMMARY_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn groupsummary_builtin(
    table: Value,
    groupvars: Value,
    method: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let table = gather_if_needed_async(&table)
        .await
        .map_err(map_control_flow)?;
    let groupvars = gather_if_needed_async(&groupvars)
        .await
        .map_err(map_control_flow)?;
    let method = gather_if_needed_async(&method)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    groupsummary_impl(table, groupvars, method, rest)
}

#[runtime_builtin(
    name = "table.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    let object = into_table_object(obj, "table.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => table_member_get(&object, &payload),
        OBJECT_INDEX_PAREN => table_paren_get(&object, &payload),
        OBJECT_INDEX_BRACE => table_brace_get(&object, &payload),
        other => Err(invalid_index(format!(
            "table.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "table.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn table_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_table_object(obj, "table.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "table member")?;
            table_member_set(&mut object, &field, rhs)?;
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN => table_paren_assign(object, &payload, rhs),
        OBJECT_INDEX_BRACE => table_brace_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "table.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    let object = into_dictionary_object(obj, "dictionary.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            object
                .properties
                .get(&field)
                .cloned()
                .ok_or_else(|| invalid_variable(format!("dictionary: unknown property '{field}'")))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_lookup(&object, &payload),
        other => Err(invalid_index(format!(
            "dictionary.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table"
)]
async fn dictionary_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_dictionary_object(obj, "dictionary.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            if field != "Keys" && field != "Values" {
                return Err(invalid_variable(format!(
                    "dictionary: unknown property '{field}'"
                )));
            }
            object.properties.insert(field, rhs);
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "dictionary.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(invalid_argument(format!(
            "table: {context} must be a string scalar or character vector"
        ))),
    }
}

fn bool_scalar(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(value) => Ok(value.to_i64() != 0),
        Value::Num(value) if value.is_finite() => Ok(*value != 0.0),
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let text = scalar_text(value, context)?;
            match text.to_ascii_lowercase().as_str() {
                "true" | "on" | "yes" => Ok(true),
                "false" | "off" | "no" => Ok(false),
                _ => Err(invalid_argument(format!(
                    "table: {context} must be logical"
                ))),
            }
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be logical"
        ))),
    }
}

fn nonnegative_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) if value.to_i64() >= 0 => Ok(value.to_i64() as usize),
        Value::Num(value)
            if value.is_finite()
                && *value >= 0.0
                && (value.round() - value).abs() <= f64::EPSILON =>
        {
            Ok(value.round() as usize)
        }
        _ => Err(invalid_argument(format!(
            "table: {context} must be a non-negative integer"
        ))),
    }
}

fn positive_usize(value: &Value, context: &str) -> BuiltinResult<usize> {
    let value = nonnegative_usize(value, context)?;
    if value == 0 {
        return Err(invalid_argument(format!(
            "table: {context} must be a positive integer"
        )));
    }
    Ok(value)
}

fn option_value_is_empty(value: &Value) -> bool {
    match value {
        Value::String(text) => text.trim().is_empty(),
        Value::CharArray(array) => {
            array.data.is_empty()
                || (array.rows == 1 && array.data.iter().all(|ch| ch.is_whitespace()))
        }
        Value::StringArray(array) => {
            array.data.is_empty() || (array.data.len() == 1 && array.data[0].trim().is_empty())
        }
        Value::Cell(cell) => {
            cell.data.is_empty() || cell.data.iter().all(|handle| option_value_is_empty(handle))
        }
        _ => false,
    }
}

fn string_list(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(ca) if ca.rows == 1 => Ok(vec![ca.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => {
            let mut out = Vec::with_capacity(cell.data.len());
            for handle in &cell.data {
                let value = handle;
                out.extend(string_list(value)?);
            }
            Ok(out)
        }
        _ => Err(invalid_argument(
            "table: expected string, string array, character vector, or cellstr",
        )),
    }
}

fn optional_raw_variable_name_list(value: &Value) -> BuiltinResult<Option<Vec<String>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        raw_variable_name_list(value).map(Some)
    }
}

fn raw_variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    let names = string_list(value)?;
    if names.is_empty() {
        return Err(invalid_variable("table: variable names must not be empty"));
    }
    Ok(names)
}

fn variable_name_list(value: &Value) -> BuiltinResult<Vec<String>> {
    raw_variable_name_list(value).map(make_unique_variable_names)
}

fn optional_variable_type_list(value: &Value) -> BuiltinResult<Option<Vec<ImportVariableType>>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        variable_type_list(value).map(Some)
    }
}

fn variable_type_list(value: &Value) -> BuiltinResult<Vec<ImportVariableType>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::parse(raw))
        .collect()
}

fn variable_type_names(value: &Value) -> BuiltinResult<Vec<String>> {
    string_list(value)?
        .iter()
        .map(|raw| ImportVariableType::canonical_label(raw))
        .collect()
}

fn optional_range_spec(value: &Value) -> BuiltinResult<Option<RangeSpec>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        RangeSpec::parse(value).map(Some)
    }
}

fn optional_sheet_selector(value: &Value) -> BuiltinResult<Option<SheetSelector>> {
    if option_value_is_empty(value) {
        Ok(None)
    } else {
        SheetSelector::parse(value).map(Some)
    }
}

fn generated_variable_names(count: usize) -> Vec<String> {
    (1..=count).map(|idx| format!("Var{idx}")).collect()
}

fn make_unique_variable_names(names: Vec<String>) -> Vec<String> {
    make_unique_names(
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| make_valid_variable_name(&name, idx + 1))
            .collect(),
    )
}

fn make_unique_names(names: Vec<String>) -> Vec<String> {
    let mut used = HashSet::new();
    let mut out = Vec::with_capacity(names.len());
    for (idx, name) in names.into_iter().enumerate() {
        let base = if name.trim().is_empty() {
            format!("Var{}", idx + 1)
        } else {
            name.trim().to_string()
        };
        let mut candidate = base.clone();
        let mut suffix = 1usize;
        while used.contains(&candidate.to_ascii_lowercase()) {
            suffix += 1;
            candidate = format!("{base}_{suffix}");
        }
        used.insert(candidate.to_ascii_lowercase());
        out.push(candidate);
    }
    out
}

fn make_valid_variable_name(raw: &str, fallback_index: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in raw.trim().chars().enumerate() {
        if (idx == 0 && (ch.is_ascii_alphabetic() || ch == '_'))
            || (idx > 0 && (ch.is_ascii_alphanumeric() || ch == '_'))
        {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || !out.chars().next().unwrap().is_ascii_alphabetic() {
        format!("Var{fallback_index}")
    } else {
        out
    }
}

#[cfg(test)]
mod tests;
