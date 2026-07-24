use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "readtable",
    category = "io/tabular",
    summary = "Import tabular text or spreadsheet data into a table.",
    keywords = "readtable,table,csv,tsv,xlsx,xls,ods,spreadsheet,VariableNames,RowNames,Sheet,Range",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::READTABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readtable_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    name = "parquetread",
    category = "io/tabular",
    summary = "Read Parquet columnar data into a table.",
    keywords = "parquetread,parquet,table,SelectedVariableNames,RowGroups,OutputType",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::PARQUETREAD_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetread_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ParquetReadOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    read_parquet_table(&resolved, &options).await
}

#[runtime_builtin(
    name = "parquetinfo",
    category = "io/tabular",
    summary = "Inspect Parquet file schema and row-group metadata.",
    keywords = "parquetinfo,parquet,schema,row groups,metadata",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::PARQUETINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquetinfo_builtin(path: Value) -> BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let resolved = resolve_path(&path_value)?;
    parquet_file_info(&resolved).await
}

#[runtime_builtin(
    name = "spreadsheetImportOptions",
    category = "io/tabular",
    summary = "Create spreadsheet import options for readtable.",
    keywords = "spreadsheetImportOptions,readtable,spreadsheet,xlsx,xls,DataRange,VariableTypes,VariableNames,NumVariables",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::SPREADSHEET_IMPORT_OPTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn spreadsheet_import_options_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn detect_import_options_builtin(
    path: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let path_value = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let args = gather_values(&rest).await?;
    let options = ReadTableOptions::parse(&args)?;
    let resolved = resolve_path(&path_value)?;
    detect_import_options_from_file(&resolved, &options).await
}

#[runtime_builtin(
    name = "writetable",
    category = "io/tabular",
    summary = "Write a table to a delimited text file.",
    keywords = "writetable,table,csv,delimited text,VariableNames",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_WRITE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetimetable_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn readcell_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let path = resolve_path(&path)?;
    let options = ReadTableOptions::parse(&rest)?;
    read_cell_from_file(&path, &options).await
}
