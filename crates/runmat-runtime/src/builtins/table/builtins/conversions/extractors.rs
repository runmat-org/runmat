use super::*;
use runmat_builtins::TABLE2STRUCT_INTEGER_TO_SCALAR_EXTENSION;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table2struct",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2struct_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.chunks_exact(2).any(|pair| {
        scalar_text(&pair[0], "table2struct option")
            .is_ok_and(|name| name.eq_ignore_ascii_case("ToScalar"))
            && crate::builtins::common::validation::value_has_native_integer_class(&pair[1])
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TABLE2STRUCT_INTEGER_TO_SCALAR_EXTENSION,
            "table2struct",
        )?;
    }
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
        rows.push(st);
    }
    runmat_value::StructArray::normalize(names, rows, vec![height, 1]).map_err(invalid_variable)
}

#[runtime_builtin(
    name = "table2array",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2array_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2array")?;
    table_brace_get(&object, &colon_colon_payload())
}

#[runtime_builtin(
    name = "table2cell",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table2cell_builtin(value: Value) -> BuiltinResult<Value> {
    let host = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let object = into_table_object(host, "table2cell")?;
    table_to_cell_array(&object)
}
