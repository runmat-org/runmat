use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "head",
    category = "table",
    summary = "Return the first rows of a table, timetable, or array.",
    keywords = "head,table,timetable,preview,rows",
    accel = "cpu",
    descriptor(crate::builtins::table::TABLE_COMPAT_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn head_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
