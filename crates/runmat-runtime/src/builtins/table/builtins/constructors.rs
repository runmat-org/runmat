use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table",
    category = "table",
    summary = "Create a table from named column variables.",
    keywords = "table,VariableNames,RowNames,Properties",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::table::TABLE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    name = "categorical",
    category = "table",
    summary = "Create a categorical array.",
    keywords = "categorical,categories,ordinal,table",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn categorical_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    categorical_from_args(args)
}

#[runtime_builtin(
    name = "ordinal",
    category = "table",
    summary = "Create an ordinal categorical array.",
    keywords = "ordinal,categorical,categories,statistics",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn ordinal_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_table_class_registered();
    let args = gather_values(&args).await?;
    ordinal_from_args(args)
}

#[runtime_builtin(
    name = "dictionary",
    category = "table",
    summary = "Create a key-value dictionary object.",
    keywords = "dictionary,containers.Map,key,value,map",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn timerange_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn vartype_builtin(value: Value) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn rowfilter_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    name = "arrayDatastore",
    category = "io/tabular",
    summary = "Create an array datastore descriptor.",
    keywords = "arrayDatastore,datastore,array,data",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn array_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn parquet_datastore_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn uitable_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
