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
mod names;
mod object;
mod parsing;
mod registry;

#[cfg(test)]
use builtins::*;
pub use metadata::*;
pub use registry::ensure_table_class_registered;

use containers::*;
pub(crate) use display::categorical_label_at;
use display::format_key_number;
pub use display::{table_display_text, table_summary_text};
use import::*;
use names::*;
use object::*;
pub use object::{
    is_table_value, is_tabular_object, sortrows_table, table_from_columns, table_height,
    table_variable_names_from_object, table_variables, table_width,
};
pub(crate) use object::{
    select_rows, selected_row_names, table_from_columns_like, value_row_count,
};
use parsing::*;
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

#[cfg(test)]
mod tests;
