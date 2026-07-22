use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "height",
    category = "table",
    summary = "Return the number of rows in a table.",
    keywords = "height,table,rows",
    descriptor(crate::builtins::table::HEIGHT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn height_builtin(value: Value) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn width_builtin(value: Value) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istable_builtin(value: Value) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istimetable_builtin(value: Value) -> BuiltinResult<Value> {
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
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn iscategorical_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object) if object.is_class(CATEGORICAL_CLASS)
    )))
}

#[runtime_builtin(
    name = "isordinal",
    category = "table",
    summary = "Return true for ordinal categorical arrays.",
    keywords = "isordinal,ordinal,categorical,predicate",
    descriptor(crate::builtins::table::TABLE_PREDICATE_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn isordinal_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    Ok(Value::Bool(matches!(
        host,
        Value::Object(ref object)
            if object.is_class(CATEGORICAL_CLASS)
                && matches!(object.properties.get("Ordinal"), Some(Value::Bool(true)))
    )))
}
