use super::*;

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
        .try_map_owned(
            |sequence| {
                let values = sequence.resolve(
                    runmat_types::SequenceUse::RequireSingle,
                    crate::sequence::SequenceResolutionContext::default(),
                )?;
                crate::sequence::into_first_output(values)?
                    .ok_or_else(|| RuntimeError::from("member read did not produce one value"))
            },
            |error| error.into_runtime_error(),
        )
}

pub async fn read_member_sequence_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    let work =
        read_member_sequence_in_scope(context, base, field, allow_init, caller_function_name);
    match context {
        Some(context) => context.scope(work).await,
        None => work.await,
    }
}

async fn read_member_sequence_in_scope(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValueSequence, RuntimeError> {
    let _source_root = RootedValueLease::for_values(std::iter::once(&base))
        .map_err(|error| error.into_runtime_error())?;
    match base {
        Value::StructArray(array) => {
            RootedValueSequence::new(crate::aggregate::structure::gather_member(array, &field)?)
                .map_err(|error| error.into_runtime_error())
        }
        Value::ObjectArray(array) => {
            let roots = crate::call::arguments::roots::ExpansionRoots::new()?;
            roots.retain_values(array.data());
            let mut values = Vec::with_capacity(array.len());
            for value in array.into_data() {
                let selected = Box::pin(read_member_sequence_with_context(
                    context,
                    value,
                    field.clone(),
                    allow_init,
                    caller_function_name,
                ))
                .await?
                .try_map_owned(
                    |sequence| {
                        let selected = sequence.resolve(
                            runmat_types::SequenceUse::RequireSingle,
                            crate::sequence::SequenceResolutionContext::default(),
                        )?;
                        Ok::<_, RuntimeError>(
                            crate::sequence::into_first_output(selected)?
                                .expect("single sequence resolution"),
                        )
                    },
                    |error| error.into_runtime_error(),
                )?;
                selected.with_owned(|selected| {
                    roots.retain_values(std::slice::from_ref(&selected));
                    values.push(selected);
                });
            }
            RootedValueSequence::new(
                crate::sequence::ValueSequence::comma_separated(values)
                    .map_err(crate::sequence::sequence_error_to_runtime)?,
            )
            .map_err(|error| error.into_runtime_error())
        }
        base => {
            load_scalar_member_with_context(context, base, field, allow_init, caller_function_name)
                .await
                .and_then(|value| {
                    value.try_map_owned(
                        |value| {
                            crate::sequence::ValueSequence::single(value)
                                .map_err(crate::sequence::sequence_error_to_runtime)
                        },
                        |error| error.into_runtime_error(),
                    )
                })
        }
    }
}

async fn load_scalar_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<RootedValue, RuntimeError> {
    if let Some(result) = context
        .and_then(|context| crate::parallel::introspection::load_member(context, &base, &field))
    {
        return RootedValue::new(result?).map_err(|error| error.into_runtime_error());
    }
    match base {
        Value::Object(obj) => {
            let selected = crate::class_registry::checked_receiver_projection(&obj)?;
            let lookup_property = |name: &runmat_types::MemberName| match &selected {
                Some(projection) => projection.receiver_property(&obj.class_name, name),
                None => crate::class_registry::lookup_property(&obj.class_name, name).map(
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
                    caller_class_for_function(caller_function_name).as_ref(),
                ),
                None => access_permitted(owner, &access, caller_function_name),
            };
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let Some(method) = crate::object::protocol::resolve_object_protocol(
                &base,
                crate::object::protocol::ObjectProtocol::Subsref,
                &access,
            )? {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
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
                            field, obj.class_name, field
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
                                &obj.class_name,
                                &runmat_types::MethodName::from(getter.as_str()),
                            )
                            .is_some()
                        },
                        |projection| {
                            projection.receiver_declares_method(
                                &obj.class_name,
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
            } else if let Some(v) = obj.properties.get(&field) {
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
                    if let Some(vb) = obj.properties.get(&backing) {
                        return RootedValue::new(vb.clone())
                            .map_err(|error| error.into_runtime_error());
                    }
                }
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else if selected.is_some()
                || crate::class_registry::get_class(&obj.class_name).is_some()
            {
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
            }?;
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
        }
        Value::HandleObject(handle) => {
            let base = Value::HandleObject(handle.clone());
            if super::graphics::is_graphics_receiver_handle(&handle) {
                return super::graphics::load_graphics_member(context, base, &field).await;
            }
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let Some(method) = crate::object::protocol::resolve_object_protocol(
                &base,
                crate::object::protocol::ObjectProtocol::Subsref,
                &access,
            )? {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
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
        Value::Foreign(reference) => crate::foreign::load_foreign_member(reference, field).await,
        Value::ClassRef(cls) => {
            RootedValue::new(load_static_member(&cls, &field, caller_function_name)?)
                .map_err(|error| error.into_runtime_error())
        }
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
            if let Some(v) = st.fields.get(&field) {
                RootedValue::new(v.clone()).map_err(|error| error.into_runtime_error())
            } else if allow_init {
                RootedValue::new(Value::Struct(StructValue::new()))
                    .map_err(|error| error.into_runtime_error())
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
                        .map(|s| Value::String(s.clone()))
                        .collect();
                    let rows = values.len();
                    let cell = runmat_value::CellArray::new(values, rows, 1)
                        .map_err(|e| format!("MException.stack: {e}"))?;
                    Value::Cell(cell)
                }
                None => return Err(format!("Reference to non-existent field '{}'.", field).into()),
            };
            RootedValue::new(value).map_err(|error| error.into_runtime_error())
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
