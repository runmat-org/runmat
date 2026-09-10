use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsref(obj: Value, subscript: Value) -> BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, table_read_step, None).await
}

async fn table_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_table_object(obj, "table.subsref")?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => table_member_get(&object, &payload),
        crate::object::indexing::ObjectIndexKind::Paren => table_paren_get(&object, &payload),
        crate::object::indexing::ObjectIndexKind::Brace => table_brace_get(&object, &payload),
    }
}

#[runtime_builtin(
    name = "table.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        table_read_step,
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| invalid_index("table assignment value is missing"))?;
            table_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn table_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let mut object = into_table_object(obj, "table.subsasgn")?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "table member")?;
            table_member_set(&mut object, &field, rhs)?;
            Ok(Value::Object(object))
        }
        crate::object::indexing::ObjectIndexKind::Paren => {
            table_paren_assign(object, &payload, rhs)
        }
        crate::object::indexing::ObjectIndexKind::Brace => {
            table_brace_assign(object, &payload, rhs)
        }
    }
}

#[runtime_builtin(
    name = "dictionary.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsref(obj: Value, subscript: Value) -> BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsref(obj, path, dictionary_read_step, None).await
}

async fn dictionary_read_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let object = into_dictionary_object(obj, "dictionary.subsref")?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "dictionary member")?;
            object
                .properties
                .get(&field)
                .cloned()
                .ok_or_else(|| invalid_variable(format!("dictionary: unknown property '{field}'")))
        }
        crate::object::indexing::ObjectIndexKind::Paren
        | crate::object::indexing::ObjectIndexKind::Brace => dictionary_lookup(&object, &payload),
    }
}

#[runtime_builtin(
    name = "dictionary.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsasgn(
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    crate::object::protocol::execute_owned_subsasgn(
        obj,
        path,
        vec![rhs],
        dictionary_read_step,
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| invalid_index("dictionary assignment value is missing"))?;
            dictionary_write_step(obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn dictionary_write_step(
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<Value> {
    let payload = step.selector_value()?;
    let mut object = into_dictionary_object(obj, "dictionary.subsasgn")?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "dictionary member")?;
            if field != "Keys" && field != "Values" {
                return Err(invalid_variable(format!(
                    "dictionary: unknown property '{field}'"
                )));
            }
            object.properties.insert(field, rhs);
            Ok(Value::Object(object))
        }
        crate::object::indexing::ObjectIndexKind::Paren
        | crate::object::indexing::ObjectIndexKind::Brace => {
            dictionary_assign(object, &payload, rhs)
        }
    }
}
