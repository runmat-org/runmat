use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "pivot",
    category = "table",
    summary = "Pivot or summarize table data by grouping variables.",
    keywords = "pivot,table,reshape,groupsummary",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_VARIADIC_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn pivot_builtin(
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
    name = "groupsummary",
    category = "table",
    summary = "Group table rows and compute summary statistics for data variables.",
    keywords = "groupsummary,group,table,mean,sum,count,median,min,max",
    accel = "cpu",
    descriptor(crate::builtins::table::GROUPSUMMARY_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn groupsummary_builtin(
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
