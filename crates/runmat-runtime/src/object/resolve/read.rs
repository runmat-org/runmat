use super::*;

pub async fn load_member(
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member_with_context(None, base, field, allow_init, caller_function_name).await
}

pub async fn load_member_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    let mut values =
        read_member_sequence_with_context(context, base, field, allow_init, caller_function_name)
            .await?
            .resolve(
                runmat_types::SequenceUse::RequireSingle,
                crate::sequence::SequenceResolutionContext::default(),
            )?;
    values
        .pop()
        .ok_or_else(|| RuntimeError::from("member read did not produce one value"))
}

pub async fn read_member_sequence_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    field: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<crate::sequence::ValueSequence, RuntimeError> {
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
) -> Result<crate::sequence::ValueSequence, RuntimeError> {
    match base {
        Value::StructArray(array) => crate::aggregate::structure::gather_member(array, &field),
        Value::ObjectArray(array) => {
            let roots = crate::call::arguments::roots::ExpansionRoots::new()?;
            roots.retain_values(array.data());
            let mut values = Vec::with_capacity(array.len());
            for value in array.into_data() {
                let mut selected = Box::pin(read_member_sequence_with_context(
                    context,
                    value,
                    field.clone(),
                    allow_init,
                    caller_function_name,
                ))
                .await?
                .resolve(
                    runmat_types::SequenceUse::RequireSingle,
                    crate::sequence::SequenceResolutionContext::default(),
                )?;
                roots.retain_values(&selected);
                values.push(selected.pop().expect("single sequence resolution"));
            }
            crate::sequence::ValueSequence::comma_separated(values)
                .map_err(crate::sequence::sequence_error_to_runtime)
        }
        base => {
            load_scalar_member_with_context(context, base, field, allow_init, caller_function_name)
                .await
                .and_then(|value| {
                    crate::sequence::ValueSequence::single(value)
                        .map_err(crate::sequence::sequence_error_to_runtime)
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
) -> Result<Value, RuntimeError> {
    if let Some(result) = context
        .and_then(|context| crate::parallel::introspection::load_member(context, &base, &field))
    {
        return result;
    }
    match base {
        Value::Object(obj) => {
            let base = Value::Object(obj.clone());
            let access = crate::object::protocol::ObjectAccessContext::from_legacy_function_name(
                caller_function_name,
            );
            if let crate::object::protocol::ProtocolResolution::Method(method) =
                crate::object::protocol::resolve_object_protocol(
                    &base,
                    crate::object::protocol::ObjectProtocol::Subsref,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                let resolution = crate::object::protocol::ProtocolResolution::Method(method);
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
                return require_single_sequence(outputs);
            }
            if let Some((p, owner)) = crate::class_registry::lookup_property(
                &obj.class_name,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
                if p.is_static {
                    return Err(mex(
                        "RunMat:PropertyStaticAccess",
                        &format!(
                            "Property '{}' is static; use classref('{}').{}",
                            field, obj.class_name, field
                        ),
                    ));
                }
                if !access_permitted(&owner, &p.get_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p.is_dependent {
                    let getter = crate::object_property_getter_name(&field);
                    if crate::class_registry::lookup_method(
                        &obj.class_name,
                        &runmat_types::MethodName::from(getter.as_str()),
                    )
                    .is_some()
                    {
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
                        return require_single_sequence(outputs);
                    }
                }
            }
            if let Some(v) = dynamicprops::dynamic_property_read(&obj, &field)? {
                Ok(v)
            } else if let Some(v) = obj.properties.get(&field) {
                Ok(v.clone())
            } else if let Some((p2, owner)) = crate::class_registry::lookup_property(
                &obj.class_name,
                &runmat_types::MemberName::from(field.as_str()),
            ) {
                if !access_permitted(&owner, &p2.get_access, caller_function_name) {
                    return Err(mex(
                        IDENT_PROPERTY_PRIVATE_ACCESS,
                        &format!("Property '{}' is private", field),
                    ));
                }
                if p2.is_dependent {
                    let backing = format!("{field}_backing");
                    if let Some(vb) = obj.properties.get(&backing) {
                        return Ok(vb.clone());
                    }
                }
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else if crate::class_registry::get_class(&obj.class_name).is_some() {
                Err(format!(
                    "Undefined property '{}' for class {}",
                    field, obj.class_name
                )
                .into())
            } else {
                Err(format!("Unknown class {}", obj.class_name).into())
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
                    crate::object::protocol::ObjectProtocol::Subsref,
                    &access,
                )?
            {
                let path = crate::object::indexing::ObjectSubscriptPath::single(
                    crate::object::indexing::ObjectSubscript::member(field),
                );
                let resolution = crate::object::protocol::ProtocolResolution::Method(method);
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
                return require_single_sequence(outputs);
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
        Value::ClassRef(cls) => load_static_member(&cls, &field, caller_function_name),
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
                Ok(v.clone())
            } else if allow_init {
                Ok(Value::Struct(StructValue::new()))
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
            let value = match field.as_str() {
                "identifier" => Value::String(mexn.identifier.clone()),
                "message" => Value::String(mexn.message.clone()),
                "stack" => {
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
                other => return Err(format!("Reference to non-existent field '{}'.", other).into()),
            };
            Ok(value)
        }
        _ => Err(mex("LoadMember", "LoadMember on non-object")),
    }
}

pub async fn load_member_dynamic(
    base: Value,
    name: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member(base, name, allow_init, caller_function_name).await
}

pub async fn load_member_dynamic_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    name: String,
    allow_init: bool,
    caller_function_name: Option<&str>,
) -> Result<Value, RuntimeError> {
    load_member_with_context(context, base, name, allow_init, caller_function_name).await
}
