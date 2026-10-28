use super::*;

pub async fn store_member<OnWrite>(
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member_with_scope(
        OrdinaryMemberScope,
        base,
        field,
        rhs,
        MemberCaller::named(caller_function_name),
        on_write,
    )
    .await
}

pub async fn store_member_in_context<OnWrite>(
    context: &crate::context::RuntimeContext,
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member_with_scope(
        ContextualMemberScope(context),
        base,
        field,
        rhs,
        MemberCaller::named(caller_function_name),
        on_write,
    )
    .await
}

pub(super) async fn store_member_with_context<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match context {
        Some(context) => {
            store_member_with_scope(
                ContextualMemberScope(context),
                base,
                field,
                rhs,
                MemberCaller::named(caller_function_name),
                on_write,
            )
            .await
        }
        None => {
            store_member_with_scope(
                OrdinaryMemberScope,
                base,
                field,
                rhs,
                MemberCaller::named(caller_function_name),
                on_write,
            )
            .await
        }
    }
}

async fn store_member_with_scope<'a, M, OnWrite>(
    scope: M,
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: MemberCaller<'a>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    M: MemberStoreScope<'a> + 'a,
    OnWrite: FnMut(&Value, &Value) + 'a,
{
    let context = scope.runtime();
    let roots = member_store_roots(context, &base, &rhs)?;
    let field = crate::call::argument_text::adopt_string(field, context)?;
    store_member_scoped(
        scope,
        MemberStore {
            base,
            field,
            rhs,
            dispatch: ProtocolDispatch::Resolve,
            _roots: roots,
        },
        caller_function_name,
        on_write,
    )
    .await
}

pub(crate) async fn store_member_text_with_optional_access<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller_function_name: Option<&str>,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match context {
        Some(context) => {
            store_member_text_with_caller_scope(
                ContextualMemberScope(context),
                base,
                field,
                rhs,
                match access {
                    Some(access) => MemberCaller::frame(caller_function_name, access),
                    None => MemberCaller::named(caller_function_name),
                },
                on_write,
            )
            .await
        }
        None => {
            store_member_text_with_caller_scope(
                OrdinaryMemberScope,
                base,
                field,
                rhs,
                match access {
                    Some(access) => MemberCaller::frame(caller_function_name, access),
                    None => MemberCaller::named(caller_function_name),
                },
                on_write,
            )
            .await
        }
    }
}

pub(super) fn store_member_text_with_caller_scope<'a, M, OnWrite>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller_function_name: MemberCaller<'a>,
    on_write: OnWrite,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    M: MemberStoreScope<'a> + 'a,
    OnWrite: FnMut(&Value, &Value) + 'a,
{
    store_member_text_with_dispatch_scope(
        scope,
        base,
        field,
        rhs,
        caller_function_name,
        on_write,
        ProtocolDispatch::Resolve,
    )
}

pub(super) async fn store_member_text_with_dispatch_scope<'a, M, OnWrite>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller_function_name: MemberCaller<'a>,
    on_write: OnWrite,
    dispatch: ProtocolDispatch,
) -> Result<RootedValue, RuntimeError>
where
    M: MemberStoreScope<'a> + 'a,
    OnWrite: FnMut(&Value, &Value) + 'a,
{
    let context = scope.runtime();
    let roots = member_store_roots(context, &base, &rhs)?;
    store_member_scoped(
        scope,
        MemberStore {
            base,
            field,
            rhs,
            dispatch,
            _roots: roots,
        },
        caller_function_name,
        on_write,
    )
    .await
}

struct MemberStore {
    base: Value,
    field: runmat_value::Utf8Text,
    rhs: Value,
    dispatch: ProtocolDispatch,
    _roots: RootedValueLease,
}

fn member_store_roots(
    context: Option<&crate::context::RuntimeContext>,
    base: &Value,
    rhs: &Value,
) -> Result<RootedValueLease, RuntimeError> {
    match context {
        Some(context) => RootedValueLease::for_value_slices_in_context(
            &[std::slice::from_ref(base), std::slice::from_ref(rhs)],
            context,
        ),
        None => RootedValueLease::for_values([base, rhs].into_iter())
            .map_err(|error| error.into_runtime_error()),
    }
}

fn store_member_scoped<'a, M, OnWrite>(
    scope: M,
    request: MemberStore,
    caller_function_name: MemberCaller<'a>,
    on_write: OnWrite,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    M: MemberStoreScope<'a> + 'a,
    OnWrite: FnMut(&Value, &Value) + 'a,
{
    scope.scope(store_member_in_scope(
        scope.runtime(),
        request,
        caller_function_name,
        on_write,
    ))
}

async fn store_member_in_scope<OnWrite>(
    context: Option<&crate::context::RuntimeContext>,
    request: MemberStore,
    caller_function_name: MemberCaller<'_>,
    mut on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let MemberStore {
        base,
        field,
        rhs,
        dispatch,
        _roots,
    } = request;
    match base {
        Value::Object(obj) => {
            let base = Value::Object(obj.clone());
            let access = caller_function_name.protocol_access();
            if let Some(method) =
                dispatch.resolve(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member_retained(field),
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
            // Invalid handle stores retain their established property-store
            // error domain before selected protocol authority is consulted.
            property_store::require_valid_handle(&handle)?;
            let access = caller_function_name.protocol_access();
            if let Some(method) =
                dispatch.resolve(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsasgn,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member_retained(field),
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
            crate::foreign::store_foreign_member_text(reference, field, rhs).await
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
            if let Some(oldv) = st.fields.get(field.as_str()) {
                on_write(oldv, &rhs);
            }
            let existing = st.fields.contains_key(field.as_str());
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                context,
                st.fields.iter()
                    .map(|(name, value)| (name, if name == field.as_str() { &rhs } else { value }))
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
    caller_function_name: Option<&str>,
    on_write: OnWrite,
) -> Result<RootedValue, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    store_member(base, name, rhs, caller_function_name, on_write).await
}

pub(crate) fn store_member_text_with_optional_access_scope<'a, M, OnWrite>(
    scope: M,
    base: Value,
    field: runmat_value::Utf8Text,
    rhs: Value,
    caller_function_name: Option<&'a str>,
    access: Option<&'a crate::object::protocol::ObjectAccessContext>,
    on_write: OnWrite,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    M: MemberStoreScope<'a> + 'a,
    OnWrite: FnMut(&Value, &Value) + 'a,
{
    store_member_text_with_caller_scope(
        scope,
        base,
        field,
        rhs,
        match access {
            Some(access) => MemberCaller::frame(caller_function_name, access),
            None => MemberCaller::named(caller_function_name),
        },
        on_write,
    )
}
pub fn store_member_traced_with_access<'a>(
    context: &'a crate::context::RuntimeContext,
    base: Value,
    field: String,
    rhs: Value,
    caller_function_name: Option<&'a str>,
    access: &'a crate::object::protocol::ObjectAccessContext,
) -> impl std::future::Future<Output = Result<RootedValue, RuntimeError>> + 'a {
    store_member_with_scope(
        ContextualMemberScope(context),
        base,
        field,
        rhs,
        MemberCaller::frame(caller_function_name, access),
        runmat_gc::gc_record_write,
    )
}
