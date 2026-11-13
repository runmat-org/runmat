use super::*;
use crate::execution::RootedValue;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "table.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime_for_read = runtime.clone();
    crate::object::protocol::execute_owned_subsref_with_context(
        runtime.as_ref(),
        obj,
        path,
        |obj, step| async move { table_read_step(runtime_for_read.as_ref(), obj, step).await },
        None,
    )
    .await
}

async fn table_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    let object = into_table_object(obj, "table.subsref")?;
    let value = match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => table_member_get(&object, &payload),
        crate::object::indexing::ObjectIndexKind::Paren => {
            table_paren_get_in_context(runtime, &object, &payload)
        }
        crate::object::indexing::ObjectIndexKind::Brace => table_brace_get(&object, &payload),
    }?;
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
    }
}

#[runtime_builtin(
    name = "table.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime_for_read = runtime.clone();
    let runtime_for_write = runtime.clone();
    crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime.as_ref(),
        obj,
        path,
        vec![rhs],
        |obj, step| async move { table_read_step(runtime_for_read.as_ref(), obj, step).await },
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| invalid_index("table assignment value is missing"))?;
            table_write_step(runtime_for_write.as_ref(), obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn table_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let payload = step.selector_value()?;
    let mut object = into_table_object(obj, "table.subsasgn")?;
    let value = match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let field = scalar_text(&payload, "table member")?;
            table_member_set(runtime, &mut object, &field, rhs)?;
            Ok(Value::Object(object))
        }
        crate::object::indexing::ObjectIndexKind::Paren => {
            table_paren_assign(runtime, object, &payload, rhs)
        }
        crate::object::indexing::ObjectIndexKind::Brace => {
            table_brace_assign(runtime, object, &payload, rhs)
        }
    }?;
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
    }
}

#[runtime_builtin(
    name = "dictionary.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<RootedValue> {
    dictionary_protocol_ingress(runtime.as_ref())?;
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime_for_read = runtime.clone();
    crate::object::protocol::execute_owned_subsref_with_context(
        runtime.as_ref(),
        obj,
        path,
        |obj, step| async move { dictionary_read_step(runtime_for_read.as_ref(), obj, step).await },
        None,
    )
    .await
}

async fn dictionary_read_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<RootedValue> {
    let object = into_dictionary_object(obj, "dictionary.subsref")?;
    let value =
        match step.kind() {
            crate::object::indexing::ObjectIndexKind::Member => {
                let crate::object::indexing::ObjectIndexSelector::Member(field) = step.selector()
                else {
                    return Err(invalid_index(
                        "dictionary: expected a typed member selector",
                    ));
                };
                let field = field.as_str();
                object.properties().get(field).cloned().ok_or_else(|| {
                    invalid_variable(format!("dictionary: unknown property '{field}'"))
                })
            }
            crate::object::indexing::ObjectIndexKind::Paren
            | crate::object::indexing::ObjectIndexKind::Brace => {
                return dictionary_lookup_selector_in_context(runtime, &object, step.selector());
            }
        }?;
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
    }
}

#[runtime_builtin(
    name = "dictionary.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    dictionary_protocol_ingress(runtime.as_ref())?;
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let runtime_for_read = runtime.clone();
    let runtime_for_write = runtime.clone();
    crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime.as_ref(),
        obj,
        path,
        vec![rhs],
        |obj, step| async move { dictionary_read_step(runtime_for_read.as_ref(), obj, step).await },
        |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| invalid_index("dictionary assignment value is missing"))?;
            dictionary_write_step(runtime_for_write.as_ref(), obj, step, rhs).await
        },
        None,
    )
    .await
}

async fn dictionary_write_step(
    runtime: Option<&crate::context::RuntimeContext>,
    obj: Value,
    step: crate::object::indexing::ObjectSubscript,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    let mut object = into_dictionary_object(obj, "dictionary.subsasgn")?;
    let value = match step.kind() {
        crate::object::indexing::ObjectIndexKind::Member => {
            let crate::object::indexing::ObjectIndexSelector::Member(field) = step.selector()
            else {
                return Err(invalid_index(
                    "dictionary: expected a typed member selector",
                ));
            };
            let field = field.as_str();
            if field != "Keys" && field != "Values" {
                return Err(invalid_variable(format!(
                    "dictionary: unknown property '{field}'"
                )));
            }
            crate::execution::object::replacement::replace_property(
                runtime,
                &mut object,
                field,
                rhs,
                |message, source| match runtime {
                    Some(runtime) => {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            Some(runtime),
                            "RunMat:dictionary:HostAllocation",
                            message,
                            source,
                        )
                    }
                    None => invalid_variable(format!("dictionary: {message}: {source}")),
                },
                |message| invalid_variable(format!("dictionary: {message}")),
            )?;
            Value::Object(object)
        }
        crate::object::indexing::ObjectIndexKind::Paren
        | crate::object::indexing::ObjectIndexKind::Brace => {
            return dictionary_assign_selector_in_context(runtime, object, step.selector(), rhs);
        }
    };
    match runtime {
        Some(runtime) => RootedValue::in_context(value, runtime),
        None => RootedValue::new(value).map_err(|error| error.into_runtime_error()),
    }
}
