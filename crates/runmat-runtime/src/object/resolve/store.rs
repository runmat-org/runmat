use super::*;

pub async fn store_member<OnWrite>(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member_with_context(
        None,
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        on_write,
    )
    .await
}

pub async fn store_member_in_context<OnWrite>(
    context: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member_with_context(
        Some(context),
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        on_write,
    )
    .await
}

pub(super) async fn store_member_with_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let work = store_member_in_scope(
        context,
        base,
        field,
        rhs,
        allow_init,
        caller_function_name,
        on_write,
    );
    match context {
        Some(context) => context.scope(work).await,
        None => work.await,
    }
}

async fn store_member_in_scope<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match base {
        Value::Object(obj) => {
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return match context {
                    Some(context) => {
                        crate::object::protocol::invoke_resolved_object_assignment_in_context(
                            context, &method, base, path, vec![rhs],
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_assignment(
                            &method, base, path, vec![rhs],
                        )
                        .await
                    }
                };
            }
            property_store::store_value(context, obj, field, rhs, caller_function_name, &mut on_write).await
        }
        Value::ClassRef(cls) => {
            if let Some((p, owner)) = crate::class_registry::lookup_property(
                &cls,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
                if !p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!("Property '{}' is not static", field),
                    ));
                }
                if p.is_constant {
                    return Err(mex(
                        IDENT_PROPERTY_READ_ONLY,
                        &format!("Property '{}' is constant", field),
                    ));
                }
                if !access_permitted(&owner, &p.set_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                crate::class_registry::set_static_property_value_in_owner(&owner, &field, rhs)?;
                Ok(Value::ClassRef(cls))
            } else {
                Err(format!("Unknown property '{}' on class {}", field, cls).into())
            }
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                return match context {
                    Some(context) => {
                        crate::object::protocol::invoke_resolved_object_assignment_in_context(
                            context, &method, base, path, vec![rhs],
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_assignment(
                            &method, base, path, vec![rhs],
                        )
                        .await
                    }
                };
            }
            property_store::store_handle(context, handle, field, rhs, caller_function_name).await
        }
        Value::Listener(_) => Err(crate::build_runtime_error(
            "setfield: listeners do not support direct field assignment",
        )
        .with_identifier("RunMat:setfield:NonStructAssignment")
        .build()),
        Value::Foreign(reference) => {
            crate::foreign::store_foreign_member(reference, field, rhs).await
        }
        Value::Num(0.0) if allow_init => {
            let mut st = StructValue::new();
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        base @ (Value::Num(_) | Value::Int(_)) => {
            if !is_possible_graphics_handle_value(&base) {
                return Err(mex("StoreMember", "StoreMember on non-object"));
            }
            store_graphics_member(context, base, &field, rhs)
                .await
                .map_err(|err| {
                    if is_invalid_graphics_handle_error(&err) {
                        mex("StoreMember", "StoreMember on non-object")
                    } else {
                        err
                    }
                })
        }
        Value::Struct(mut st) => {
            if let Some(oldv) = st.fields.get(&field) {
                on_write(oldv, &rhs);
            }
            st.fields.insert(field, rhs);
            Ok(Value::Struct(st))
        }
        Value::StructArray(_) => Err(mex(
            "RunMat:CommaSeparatedListAssignmentArity",
            "simple member assignment cannot target multiple structure elements; use a comma-separated destination list",
        )),
        Value::Cell(_) => Err(mex("StoreMember", "StoreMember on non-object")),
        _ => Err(mex("StoreMember", "StoreMember on non-object")),
    }
}

pub async fn store_member_dynamic<OnWrite>(
    base: Value,
    name: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member(base, name, rhs, allow_init, caller_function_name, on_write).await
}
