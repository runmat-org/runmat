pub(in crate::builtins::table) use std::cmp::Ordering;
pub(in crate::builtins::table) use std::collections::{BTreeMap, HashSet};
pub(in crate::builtins::table) use std::io::{Cursor, Read};
pub(in crate::builtins::table) use std::path::{Path, PathBuf};

pub(in crate::builtins::table) use calamine::{
    open_workbook_auto_from_rs, Data as SpreadsheetData, Reader as SpreadsheetReader,
};
pub(in crate::builtins::table) use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
pub(in crate::builtins::table) use encoding_rs::{Encoding, UTF_8};
pub(in crate::builtins::table) use runmat_filesystem::File;
pub(in crate::builtins::table) use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, NumericDType, ObjectInstance, StringArray,
    StructValue, Tensor, Value,
};

pub(in crate::builtins::table) use crate::builtins::common::fs::expand_user_path;
pub(in crate::builtins::table) use crate::{gather_if_needed_async, BuiltinResult};

pub const TABLE_CLASS: runmat_types::StaticClassIdentity = runmat_types::standard::TABLE;
pub const TIMETABLE_CLASS: runmat_types::StaticClassIdentity = runmat_types::standard::TIMETABLE;

pub(in crate::builtins::table) const CATEGORICAL_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::CATEGORICAL;
pub(in crate::builtins::table) const DICTIONARY_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DICTIONARY;
pub(in crate::builtins::table) const TIMERANGE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("timerange");
pub(in crate::builtins::table) const VARTYPE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("vartype");
pub(in crate::builtins::table) const ROWFILTER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("rowfilter");
pub(in crate::builtins::table) const ARRAY_DATASTORE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("arrayDatastore");
pub(in crate::builtins::table) const FILE_DATASTORE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("fileDatastore");
pub(in crate::builtins::table) const PARQUET_DATASTORE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("parquetDatastore");
pub(in crate::builtins::table) const UITABLE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("uitable");

pub(in crate::builtins::table) const TABLE_VARIABLES_FIELD: &str = "__table_variables";
pub(in crate::builtins::table) const TABLE_PROPERTIES_FIELD: &str = "__table_properties";
pub(in crate::builtins::table) const PROPERTIES_MEMBER: &str = "Properties";
pub(in crate::builtins::table) const ROW_TIMES: &str = "RowTimes";
pub(in crate::builtins::table) const VARIABLE_NAMES: &str = "VariableNames";
pub(in crate::builtins::table) const ROW_NAMES: &str = "RowNames";
pub(in crate::builtins::table) const DIMENSION_NAMES: &str = "DimensionNames";
pub(in crate::builtins::table) const VARIABLE_UNITS: &str = "VariableUnits";
pub(in crate::builtins::table) const VARIABLE_DESCRIPTIONS: &str = "VariableDescriptions";
pub(in crate::builtins::table) const DESCRIPTION: &str = "Description";
pub(in crate::builtins::table) const USER_DATA: &str = "UserData";
pub(in crate::builtins::table) const DEFAULT_ROW_DIM_NAME: &str = "Rows";
pub(in crate::builtins::table) const DEFAULT_VARIABLE_DIM_NAME: &str = "Variables";
