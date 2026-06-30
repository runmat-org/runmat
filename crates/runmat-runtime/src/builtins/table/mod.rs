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

use crate::builtins::common::fs::expand_user_path;
use crate::{
    gather_if_needed_async, BuiltinResult, OBJECT_INDEX_BRACE, OBJECT_INDEX_MEMBER,
    OBJECT_INDEX_PAREN,
};

pub(crate) mod builtins;
mod containers;
mod display;
mod import;
mod metadata;
mod object;
mod registry;

#[cfg(test)]
use builtins::*;
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
