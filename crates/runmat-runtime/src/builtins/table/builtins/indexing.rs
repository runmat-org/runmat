use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<Value> {
    let object = into_table_object(obj, "table.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => table_member_get(&object, &payload),
        OBJECT_INDEX_PAREN => table_paren_get(&object, &payload),
        OBJECT_INDEX_BRACE => table_brace_get(&object, &payload),
        other => Err(invalid_index(format!(
            "table.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "table.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_table_object(obj, "table.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "table member")?;
            table_member_set(&mut object, &field, rhs)?;
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN => table_paren_assign(object, &payload, rhs),
        OBJECT_INDEX_BRACE => table_brace_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "table.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsref",
    descriptor(crate::builtins::table::TABLE_SUBSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<Value> {
    let object = into_dictionary_object(obj, "dictionary.subsref")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            object
                .properties
                .get(&field)
                .cloned()
                .ok_or_else(|| invalid_variable(format!("dictionary: unknown property '{field}'")))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_lookup(&object, &payload),
        other => Err(invalid_index(format!(
            "dictionary.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runtime_builtin(
    name = "dictionary.subsasgn",
    descriptor(crate::builtins::table::TABLE_SUBSASGN_DESCRIPTOR),
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> BuiltinResult<Value> {
    let mut object = into_dictionary_object(obj, "dictionary.subsasgn")?;
    match kind.as_str() {
        OBJECT_INDEX_MEMBER => {
            let field = scalar_text(&payload, "dictionary member")?;
            if field != "Keys" && field != "Values" {
                return Err(invalid_variable(format!(
                    "dictionary: unknown property '{field}'"
                )));
            }
            object.properties.insert(field, rhs);
            Ok(Value::Object(object))
        }
        OBJECT_INDEX_PAREN | OBJECT_INDEX_BRACE => dictionary_assign(object, &payload, rhs),
        other => Err(invalid_index(format!(
            "dictionary.subsasgn: unsupported indexing kind '{other}'"
        ))),
    }
}
