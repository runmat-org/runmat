use super::*;

pub async fn store_member<OnWrite>(
    base: Value,
    field: String,
    rhs: Value,
    allow_init: bool,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
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
) -> Result<RootedValue, RuntimeError>
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
) -> Result<RootedValue, RuntimeError>
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
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let _inputs_root = match context {
        Some(context) => RootedValueLease::for_value_slices_in_context(
            &[std::slice::from_ref(&base), std::slice::from_ref(&rhs)],
            context,
        )?,
        None => RootedValueLease::for_values([&base, &rhs].into_iter())
            .map_err(|error| error.into_runtime_error())?,
    };
    match base {
        Value::Object(obj) => {
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let Some(method) =
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
                RootedValue::new(Value::ClassRef(cls))
                    .map_err(|error| error.into_runtime_error())
            } else {
                Err(format!("Unknown property '{}' on class {}", field, cls).into())
            }
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            if super::graphics::is_graphics_receiver_handle(&handle) {
                return super::graphics::store_graphics_member(context, base, &field, rhs).await;
            }
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let Some(method) =
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
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                context, std::iter::once((field.as_str(), &rhs)),
            )
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
        Value::Struct(st) => {
            if let Some(oldv) = st.fields.get(&field) {
                on_write(oldv, &rhs);
            }
            let existing = st.fields.contains_key(&field);
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                context,
                st.fields.iter()
                    .map(|(name, value)| (name, if name == field { &rhs } else { value }))
                    .chain((!existing).then_some((field.as_str(), &rhs))),
            )
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
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member(base, name, rhs, allow_init, caller_function_name, on_write).await
}
