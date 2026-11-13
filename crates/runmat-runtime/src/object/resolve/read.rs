use super::*;
#[path = "read/entry.rs"]
mod entry;
pub(crate) use entry::read_member_sequence_text_with_optional_access;
pub use entry::*;

pub async fn load_member(
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    load_member_with_context(None, base, field, allow_init, caller_function_name).await
}

pub async fn load_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    read_member_sequence_with_context(context, base, field, allow_init, caller_function_name)
        .await?
        .try_into_single()
}

async fn read_member_sequence_with_caller(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: MemberCaller<'_>,
) -> Result<RootedValueSequence, RuntimeError> {
    let roots = member_read_roots(context, &base)?;
    let field = crate::call::argument_text::adopt_string(field, context)?;
    read_member_sequence_scoped(
        context,
        MemberRead {
            base,
            field,
            allow_init,
            dispatch: ProtocolDispatch::Resolve,
            _roots: roots,
        },
        caller_function_name,
    )
    .await
}

pub(super) fn read_member_sequence_text_with_caller<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: MemberCaller<'a>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    read_member_sequence_text_with_dispatch(
        context,
        base,
        field,
        allow_init,
        caller_function_name,
        ProtocolDispatch::Resolve,
    )
}

pub(super) async fn read_member_sequence_text_with_dispatch(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: MemberCaller<'_>,
    dispatch: ProtocolDispatch,
) -> Result<RootedValueSequence, RuntimeError> {
    let roots = member_read_roots(context, &base)?;
    read_member_sequence_scoped(
        context,
        MemberRead {
            base,
            field,
            allow_init,
            dispatch,
            _roots: roots,
        },
        caller_function_name,
    )
    .await
}

struct MemberRead {
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    dispatch: ProtocolDispatch,
    _roots: RootedValueLease,
}

fn member_read_roots(
    context: Option<&crate::context::RuntimeContext>,
    base: &Value,
) -> Result<RootedValueLease, RuntimeError> {
    crate::execution::structure::replacement::lease(base, context)
}

fn read_member_sequence_scoped<'a>(
    context: Option<&'a crate::context::RuntimeContext>,
    request: MemberRead,
    caller_function_name: MemberCaller<'a>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a {
    // The callers already defer and retain roots before reaching this scope
    // boundary. Return the same scoped child, without a forwarding poll frame.
    let work = read_member_sequence_in_scope(context, request, caller_function_name);
    match context {
        Some(context) => futures::future::Either::Left(context.scope(work)),
        None => futures::future::Either::Right(work),
    }
}

async fn read_member_sequence_in_scope(
    context: Option<&crate::context::RuntimeContext>,
    request: MemberRead,
    caller_function_name: MemberCaller<'_>,
) -> Result<RootedValueSequence, RuntimeError> {
    let MemberRead {
        base,
        field,
        allow_init,
        dispatch,
        _roots,
    } = request;
    match base {
        Value::StructArray(array) => {
            crate::aggregate::structure::gather_member(context, array, &field)
        }
        Value::ObjectArray(array) => {
            super::read_array::gather_object_members(
                context,
                array,
                field,
                allow_init,
                caller_function_name,
                dispatch,
            )
            .await
        }
        base => load_scalar_member_with_context(
            context,
            base,
            field,
            allow_init,
            caller_function_name,
            dispatch,
        )
        .await
        .and_then(RootedValue::try_into_sequence),
    }
}

async fn load_scalar_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: runmat_value::Utf8Text,
    allow_init: bool,
    caller_function_name: MemberCaller<'_>,
    dispatch: ProtocolDispatch,
) -> Result<RootedValue, RuntimeError> {
    if let Some(result) = context
        .and_then(|context| crate::parallel::introspection::load_member(context, &base, &field))
    {
        return RootedValue::in_optional_context(result?, context);
    }
    match base {
        Value::Object(obj) => {
            let selected = crate::class_registry::checked_receiver_projection(&obj)?;
            let lookup_property = |name: &runmat_types::MemberName| match &selected {
                Some(projection) => projection.receiver_property(obj.class_name(), name),
                None => crate::class_registry::lookup_property(obj.class_name(), name).map(
                    |(property, owner)| {
                        (
                            crate::class_registry::receiver_property_declaration(property),
                            owner,
                        )
                    },
                ),
            };
            let property_accessible = |owner: &ClassIdentity, access| match &selected {
                Some(projection) => projection.receiver_member_accessible(
                    owner,
                    access,
                    caller_function_name.class().as_ref(),
                ),
                None => access_permitted(owner, &access, caller_function_name),
            };
            let base = Value::Object(obj.clone());
            let access = caller_function_name.protocol_access();
            if let Some(method) = dispatch.resolve(
                &base,
                crate::object::protocol::ObjectProtocol::Subsref,
                &access,
            )? {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member_retained(field),
                );
                let resolution = Some(method);
                let outputs = match context {
                    Some(context) => {
                        crate::object::protocol::invoke_resolved_object_protocol_in_context(
                            context,
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_protocol(
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await
                    }
                }?;
                return require_single_rooted_sequence(outputs);
            }
            if let Some((p, owner)) =
                lookup_property(&runmat_types::MemberName::from(field.as_str()))
            {
                if p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!(
                            "Property '{}' is static; use classref('{}').{}",
                            field,
                            obj.class_name(),
                            field
                        ),
                    ));
                }
                if !property_accessible(&owner, p.get_access) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p.is_dependent {
                    let getter = crate::object_property_getter_name(&field);
                    if selected.as_ref().map_or_else(
                        || {
                            crate::class_registry::lookup_method(
                                obj.class_name(),
                                &runmat_types::MethodName::from(getter.as_str()),
                            )
                            .is_some()
                        },
                        |projection| {
                            projection.receiver_declares_method(
                                obj.class_name(),
                                &runmat_types::MethodName::from(getter.as_str()),
                            )
                        },
                    ) {
                        let outputs = match context {
                            Some(context) => {
                                crate::object::dispatch::call_object_property_getter_with_outputs_in_context(
                                    context,
                                    Value::Object(obj.clone()),
                                    &field,
                                    1,
                                )
                                .await
                            }
                            None => {
                                call_object_property_getter_with_outputs(
                                    Value::Object(obj.clone()),
                                    &field,
                                    1,
                                )
                                .await
                            }
                        }?;
                        return require_single_rooted_sequence(outputs);
                    }
                }
            }
            let value = if let Some(v) = dynamicprops::dynamic_property_read(&obj, &field)? {
                Ok::<Value, crate::RuntimeError>(v)
            } else if let Some(v) = obj.property(&field) {
                Ok(v.clone())
            } else if let Some((p2, owner)) =
                lookup_property(&runmat_types::MemberName::from(field.as_str()))
            {
                if !property_accessible(&owner, p2.get_access) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p2.is_dependent {
                    let backing = format!("{field}_backing");
                    if let Some(vb) = obj.property(&backing) {
                        return RootedValue::in_optional_context(vb.clone(), context);
                    }
                }
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field,
                    obj.class_name()
                )
                .into())
            } else if selected.is_some()
                || crate::class_registry::get_class(obj.class_name()).is_some()
            {
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field,
                    obj.class_name()
                )
                .into())
            } else {
                Err(format!("Unknown class {}", obj.class_name()).into())
            }?;
            RootedValue::in_optional_context(value, context)
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            if super::graphics::is_graphics_receiver_handle(&handle) {
                return super::graphics::load_graphics_member(context, base, &field).await;
            }
            let access = caller_function_name.protocol_access();
            if let Some(method) = dispatch.resolve(
                &base,
                crate::object::protocol::ObjectProtocol::Subsref,
                &access,
            )? {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member_retained(field),
                );
                let resolution = Some(method);
                let outputs = match context {
                    Some(context) => {
                        crate::object::protocol::invoke_resolved_object_protocol_in_context(
                            context,
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await
                    }
                    None => {
                        crate::object::protocol::invoke_resolved_object_protocol(
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await
                    }
                }?;
                return require_single_rooted_sequence(outputs);
            }
            crate::builtins::structs::core::getfield::get_member_value_in_context(
                context,
                Value::HandleObject(handle),
                &field,
            )
            .await
        }
        Value::Listener(listener) => {
            crate::builtins::structs::core::getfield::get_member_value_in_context(
                context,
                Value::Listener(listener),
                &field,
            )
            .await
        }
        Value::Foreign(reference) => {
            crate::foreign::load_foreign_member_text(reference, field).await
        }
        Value::ClassRef(cls) => RootedValue::in_optional_context(
            super::static_member::load_static_member_with_caller(
                &cls,
                &field,
                caller_function_name,
            )?,
            context,
        ),
        base @ (Value::Num(_) | Value::Int(_)) => {
            if !is_possible_graphics_handle_value(&base) {
                return Err(mex("LoadMember", "LoadMember on non-object"));
            }
            load_graphics_member(context, base, &field)
                .await
                .map_err(|err| {
                    if is_invalid_graphics_handle_error(&err) {
                        mex("LoadMember", "LoadMember on non-object")
                    } else {
                        err
                    }
                })
        }
        Value::Struct(st) => {
            if let Some(v) = st.fields.get(field.as_str()) {
                RootedValue::in_optional_context(v.clone(), context)
            } else if allow_init {
                RootedValue::in_optional_context(Value::Struct(StructValue::new()), context)
            } else {
                Err(format!("Undefined field '{}'", field).into())
            }
        }
        Value::StructArray(_) | Value::ObjectArray(_) => Err(mex(
            "MemberSequenceInvariant",
            "aggregate member reads must pass through sequence resolution",
        )),
        Value::Cell(_) => Err(mex("LoadMember", "LoadMember on non-object")),
        Value::MException(mexn) => {
            let value = match runmat_types::ExceptionProperty::from_name(field.as_str()) {
                Some(runmat_types::ExceptionProperty::Identifier) => {
                    exception_text_value(&mexn.identifier)
                }
                Some(runmat_types::ExceptionProperty::Message) => {
                    exception_text_value(&mexn.message)
                }
                Some(runmat_types::ExceptionProperty::Stack) => {
                    let values: Vec<Value> = mexn
                        .stack
                        .iter()
                        .map(|s| Value::string(s.clone()))
                        .collect();
                    let rows = values.len();
                    let cell = runmat_value::CellArray::new(values, rows, 1)
                        .map_err(|e| format!("MException.stack: {e}"))?;
                    Value::Cell(cell)
                }
                None => return Err(format!("Reference to non-existent field '{}'.", field).into()),
            };
            RootedValue::in_optional_context(value, context)
        }
        _ => Err(mex("LoadMember", "LoadMember on non-object")),
    }
}

pub async fn load_member_dynamic(
    base: Value,
    name: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    load_member(base, name, allow_init, caller_function_name).await
}

pub async fn load_member_dynamic_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    load_member_with_context(context, base, name, allow_init, caller_function_name).await
}
