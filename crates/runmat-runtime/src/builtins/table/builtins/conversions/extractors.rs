use super::*;
use runmat_builtins::TABLE2STRUCT_INTEGER_TO_SCALAR_EXTENSION;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table2struct",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins::conversions::extractors"
)]
pub(crate) async fn table2struct_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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
    let host = gather_retained_value(&value, runtime.as_ref()).await?;
    let rest = gather_retained_values(&rest, runtime.as_ref()).await?;
    let to_scalar = parse_table2struct_to_scalar(rest.view())?;
    let object = into_table_object(host, "table2struct")?;
    if to_scalar {
        return Ok(Value::Struct(table_variables(&object)?));
    }
    let height = table_height(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let mut rows = Vec::with_capacity(height);
    for row in 0..height {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        for name in &names {
            let value = variables.fields.get(name).ok_or_else(|| {
                invalid_variable(format!("table2struct: missing variable '{name}'"))
            })?;
            st.insert(name.clone(), row_value(value, row)?);
        }
        let st = st.finish();
        rows.push(st);
    }
    runmat_value::StructArray::normalize(names, rows, vec![height, 1]).map_err(invalid_variable)
}

#[runtime_builtin(
    name = "table2array",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins::conversions::extractors"
)]
pub(crate) async fn table2array_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let host = gather_retained_value(&value, runtime.as_ref()).await?;
    let object = into_table_object(host, "table2array")?;
    let variables = table_variables(&object)?;
    if !variables.fields.is_empty()
        && variables
            .fields
            .values()
            .all(|value| matches!(value, Value::Cell(_)))
    {
        return table_to_cell_array(&object);
    }
    table_brace_get(&object, &colon_colon_payload())
}

#[runtime_builtin(
    name = "table2cell",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins::conversions::extractors"
)]
pub(crate) async fn table2cell_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    let host = gather_retained_value(&value, runtime.as_ref()).await?;
    let object = into_table_object(host, "table2cell")?;
    table_to_cell_array(&object)
}
