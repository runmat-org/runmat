use super::{DYNAMICPROPS_CLASS, DYNAMIC_PROPERTY_CLASS, TARGET_FIELD, VALID_FIELD};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, DYNAMIC_ERROR_INVALID_METADATA, DYNAMIC_ERROR_INVALID_NAME,
    DYNAMIC_ERROR_INVALID_TARGET, DYNAMIC_ERROR_UNSUPPORTED_METADATA,
};
use runmat_types::MemberAccess;
use runmat_value::{DynamicPropertyDef, HandleRef, ObjectInstance, Tensor, Value};

pub(super) fn dynamic_error(
    builtin: impl AsRef<str>,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin.as_ref())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn dynamic_error_with_source(
    builtin: impl AsRef<str>,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin.as_ref())
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub fn empty_metadata_result() -> Value {
    Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("empty tensor"))
}

pub(super) fn validate_property_name(name: &str, builtin: &'static str) -> BuiltinResult<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(dynamic_error(
            builtin,
            &DYNAMIC_ERROR_INVALID_NAME,
            format!("{builtin}: property name must not be empty"),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(dynamic_error(
            builtin,
            &DYNAMIC_ERROR_INVALID_NAME,
            format!("{builtin}: invalid property name '{name}'"),
        ));
    }
    if chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric())) {
        return Err(dynamic_error(
            builtin,
            &DYNAMIC_ERROR_INVALID_NAME,
            format!("{builtin}: invalid property name '{name}'"),
        ));
    }
    Ok(())
}

pub(super) fn empty_default_value() -> Value {
    empty_metadata_result()
}

pub(super) fn metadata_text(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    field: &str,
) -> BuiltinResult<runmat_value::Utf8Text> {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => {
            crate::call::argument_text::convert(value, runtime)
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::call::argument_text::convert(value, runtime)
        }
        Value::StringArray(array) if array.data().len() == 1 => {
            crate::call::argument_text::convert(value, runtime)
        }
        _ => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' requires scalar text"),
        )),
    }
}

pub(super) fn metadata_bool(value: &Value, field: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(number) if *number == 0.0 || *number == 1.0 => Ok(*number != 0.0),
        Value::Int(int) => match int.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
                format!("dynamic property metadata field '{field}' requires logical scalar"),
            )),
        },
        _ => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' requires logical scalar"),
        )),
    }
}

pub(super) fn metadata_access(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    field: &str,
) -> BuiltinResult<MemberAccess> {
    let text = metadata_text(runtime, value, field)?;
    if text.eq_ignore_ascii_case("public") {
        Ok(MemberAccess::Public)
    } else if text.eq_ignore_ascii_case("private") {
        Ok(MemberAccess::Private)
    } else if text.eq_ignore_ascii_case("protected") {
        Ok(MemberAccess::Protected)
    } else {
        Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!(
                "dynamic property metadata field '{field}' has unsupported access '{}'",
                text.to_ascii_lowercase()
            ),
        ))
    }
}

pub(super) fn access_value(access: &MemberAccess) -> Value {
    let text = match access {
        MemberAccess::Public => "public",
        MemberAccess::Private => "private",
        MemberAccess::Protected => "protected",
    };
    Value::string(text.to_string())
}

pub(super) fn dynamic_def_to_metadata_object(
    def: &DynamicPropertyDef,
    target: Option<Value>,
    class_name: runmat_types::StaticClassIdentity,
) -> ObjectInstance {
    let mut object = runmat_value::UnboundedObjectBuilder::new(class_name);
    object.insert("Name".to_string(), Value::string(def.name.clone()));
    object.insert(
        "DefiningClass".to_string(),
        Value::ClassRef(def.defining_class.clone().into()),
    );
    object.insert("GetAccess".to_string(), access_value(&def.get_access));
    object.insert("SetAccess".to_string(), access_value(&def.set_access));
    object.insert("Dependent".to_string(), Value::Bool(def.dependent));
    object.insert("Hidden".to_string(), Value::Bool(def.hidden));
    object.insert("Transient".to_string(), Value::Bool(def.transient));
    object.insert("NonCopyable".to_string(), Value::Bool(def.non_copyable));
    object.insert("AbortSet".to_string(), Value::Bool(def.abort_set));
    object.insert("SetObservable".to_string(), Value::Bool(def.set_observable));
    object.insert("GetObservable".to_string(), Value::Bool(def.get_observable));
    object.insert(
        "Description".to_string(),
        Value::string(def.description.clone()),
    );
    object.insert(VALID_FIELD.to_string(), Value::Bool(true));
    if let Some(target) = target {
        object.insert(TARGET_FIELD.to_string(), target);
    }
    object.finish()
}

pub(super) fn dynamic_def_from_class_property(
    runtime: Option<&crate::context::RuntimeContext>,
    class_name: &runmat_types::ClassIdentity,
    name: &runmat_value::Utf8Text,
    prop: &runmat_value::SelectedReceiverPropertyDeclaration,
) -> BuiltinResult<DynamicPropertyDef> {
    use crate::context::host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    };
    let error = |source| {
        dynamic_error_with_source(
            "findprop",
            &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
            "findprop: class metadata admission failed",
            source,
        )
    };
    let bytes = runmat_value::SharedClassIdentity::requested_bytes_for_capacity(
        class_name.display_name().len(),
    )
    .map_err(|source| error(HostAllocationError::from(source)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?,
        1,
    )
    .map_err(error)?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained).map_err(error)?,
    )
    .map_err(error)?;
    let class_name = match &receipt {
        Some(receipt) => {
            let runtime = runtime.expect("managed class metadata has context");
            let permit = receipt.output_permit(runtime).map_err(error)?;
            let class =
                runmat_value::PreparedObjectClassName::prepare(class_name.display_name(), &permit)
                    .map_err(|source| {
                        dynamic_error_with_source(
                            "findprop",
                            &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
                            "findprop: class metadata preparation failed",
                            source,
                        )
                    })?
                    .into_identity();
            receipt.checkpoint(runtime).map_err(error)?;
            class
        }
        None => class_name.clone().into(),
    };
    let mut def = DynamicPropertyDef::new(String::new(), class_name);
    def.name = name.clone();
    def.get_access = prop.get_access;
    def.set_access = prop.set_access;
    def.dependent = prop.is_dependent;
    Ok(def)
}

pub(super) fn allocate_dynamic_property_handle(
    runtime: Option<&crate::context::RuntimeContext>,
    def: &DynamicPropertyDef,
    target: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let metadata =
        super::projection::metadata_object(runtime, def, Some(target), DYNAMIC_PROPERTY_CLASS)?;
    let Value::Object(object) = metadata.value() else {
        unreachable!("dynamic metadata object")
    };
    let class_name = object.class_name_owner().clone();
    metadata.with_owned(|value| {
        let handle = runmat_gc::gc_allocate_rooted(value).map_err(|source| {
            dynamic_error_with_source(
                "addprop",
                &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
                "addprop: failed to allocate metadata handle",
                source,
            )
        })?;
        crate::execution::RootedValue::in_optional_context(
            Value::HandleObject(HandleRef {
                class_name,
                target: handle.handle(),
                valid: true,
            }),
            runtime,
        )
    })
}

pub(super) fn dynamic_property_handle_from_gc(
    runtime: Option<&crate::context::RuntimeContext>,
    target: runmat_gc::GcHandle,
) -> BuiltinResult<crate::execution::RootedValue> {
    let class_name = runmat_gc::gc_with_value(&target, |value| match value {
        Value::Object(object) if object.is_class(DYNAMIC_PROPERTY_CLASS) => {
            Some(object.class_name_owner().clone())
        }
        _ => None,
    })
    .map_err(|source| {
        dynamic_error_with_source(
            "findprop",
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic metadata handle is unavailable",
            source,
        )
    })?
    .ok_or_else(|| {
        dynamic_error(
            "findprop",
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic metadata handle does not reference its metadata class",
        )
    })?;
    crate::execution::RootedValue::in_optional_context(
        Value::HandleObject(HandleRef {
            class_name,
            target,
            valid: true,
        }),
        runtime,
    )
}

pub(super) fn require_dynamicprops_target(
    class_name: &runmat_types::ClassIdentity,
) -> BuiltinResult<()> {
    if crate::class_registry::is_class_or_subclass(class_name, &DYNAMICPROPS_CLASS.owned()) {
        Ok(())
    } else {
        Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("addprop: class '{class_name}' is not a dynamicprops subclass"),
        ))
    }
}

pub(super) fn metadata_target_and_name(
    runtime: Option<&crate::context::RuntimeContext>,
    metadata: &ObjectInstance,
) -> BuiltinResult<(HandleRef, runmat_value::Utf8Text)> {
    if !metadata.class_name().is(DYNAMIC_PROPERTY_CLASS) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property operation requires matlab.metadata.DynamicProperty",
        ));
    }
    if !matches!(metadata.property(VALID_FIELD), Some(Value::Bool(true))) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata handle is invalid",
        ));
    }
    let Some(Value::HandleObject(target)) = metadata.property(TARGET_FIELD).cloned() else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata is missing its target object",
        ));
    };
    let name = metadata_text(
        runtime,
        metadata.property("Name").ok_or_else(|| {
            dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                "dynamic property metadata is missing its Name",
            )
        })?,
        "Name",
    )?;
    Ok((target, name))
}

pub fn dynamic_property_get(
    obj: &ObjectInstance,
    name: &str,
) -> Option<(DynamicPropertyDef, Value)> {
    let def = obj.dynamic_property(name)?.clone();
    let value = obj
        .property(name)
        .cloned()
        .unwrap_or_else(empty_default_value);
    Some((def, value))
}

pub fn dynamic_property_exists(obj: &ObjectInstance, name: &str) -> bool {
    obj.has_dynamic_property(name)
}

pub fn dynamic_property_assign(
    obj: &mut ObjectInstance,
    name: &str,
    value: Value,
) -> BuiltinResult<bool> {
    let Some(permitted) = dynamic_property_assignment_access(obj, name, None) else {
        return Ok(false);
    };
    if !permitted {
        return Err(dynamic_error(
            "setfield",
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("Property '{name}' is private"),
        ));
    }
    let mut replacement = obj.clone().into_unbounded_builder().map_err(|_| {
        dynamic_error(
            "setfield",
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            "dynamic property requires an admitted edit generation",
        )
    })?;
    replacement.insert(name.to_string(), value);
    *obj = replacement.finish();
    Ok(true)
}

/// Dynamic-property access policy, shared by public field assignment and the
/// canonical object store. Absence is distinct from a denied existing property.
/// The caller is the class resolved from canonical method metadata, not a
/// receiver-supplied class name.
pub(crate) fn dynamic_property_assignment_access(
    obj: &ObjectInstance,
    name: &str,
    caller: Option<&runmat_types::ClassIdentity>,
) -> Option<bool> {
    let def = obj.dynamic_property(name)?;
    Some(match def.set_access {
        MemberAccess::Public => true,
        MemberAccess::Private => caller == Some(def.defining_class.as_identity()),
        MemberAccess::Protected => caller.is_some_and(|caller| {
            crate::class_registry::is_class_or_subclass(caller, def.defining_class.as_identity())
        }),
    })
}

pub fn dynamic_property_read(obj: &ObjectInstance, name: &str) -> BuiltinResult<Option<Value>> {
    let Some((def, value)) = dynamic_property_get(obj, name) else {
        return Ok(None);
    };
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if !projection.receiver_member_accessible(
            def.defining_class.as_identity(),
            def.get_access,
            crate::dispatcher::class_access_context().as_deref(),
        ) {
            return Err(dynamic_error(
                "getfield",
                &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
                format!("Property '{name}' is not accessible"),
            ));
        }
        return Ok(Some(value));
    }
    if def.get_access == MemberAccess::Private {
        return Err(dynamic_error(
            "getfield",
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("Property '{name}' is private"),
        ));
    }
    Ok(Some(value))
}

pub fn metadata_assignment(
    runtime: Option<&crate::context::RuntimeContext>,
    metadata: &mut ObjectInstance,
    field: &str,
    value: Value,
) -> BuiltinResult<bool> {
    if !metadata.class_name().is(DYNAMIC_PROPERTY_CLASS) {
        return Ok(false);
    }
    if field == "Name" || field == "DefiningClass" {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' is read-only"),
        ));
    }
    let owner = crate::context::host_allocation::HostAllocationReceipt::capture_owner(runtime)
        .map_err(|source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:dynamicprops:Allocation",
                "dynamic metadata publication authority failed",
                source,
            )
        })?;
    let _sources = crate::execution::RootedValue::in_optional_context(
        Value::Object(metadata.clone()),
        runtime,
    )?;
    let (target, name) = metadata_target_and_name(runtime, metadata)?;
    let source = runmat_gc::gc_with_value(&target.target, |target| match target {
        Value::Object(object) => Ok(object.clone()),
        _ => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property target is not an object",
        )),
    })
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property target is invalid",
            source,
        )
    })??;
    let mut definition = source.dynamic_property(&name).cloned().ok_or_else(|| {
        dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            format!("dynamic property '{name}' no longer exists"),
        )
    })?;
    match field {
        "GetAccess" => definition.get_access = metadata_access(runtime, &value, field)?,
        "SetAccess" => definition.set_access = metadata_access(runtime, &value, field)?,
        "Dependent" => definition.dependent = metadata_bool(&value, field)?,
        "Hidden" => definition.hidden = metadata_bool(&value, field)?,
        "Transient" => definition.transient = metadata_bool(&value, field)?,
        "NonCopyable" => definition.non_copyable = metadata_bool(&value, field)?,
        "AbortSet" => definition.abort_set = metadata_bool(&value, field)?,
        "SetObservable" => definition.set_observable = metadata_bool(&value, field)?,
        "GetObservable" => definition.get_observable = metadata_bool(&value, field)?,
        "Description" => {
            definition.description = metadata_text(runtime, &value, field)?;
        }
        _ => {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
                format!("unsupported dynamic property metadata field '{field}'"),
            ))
        }
    }
    // Prepare both generations before either target is changed. Rooted metadata
    // owns the new handle edge; the original target remains rooted by _sources.
    let projected = super::projection::metadata_object(
        runtime,
        &definition,
        Some(Value::HandleObject(target.clone())),
        DYNAMIC_PROPERTY_CLASS,
    )?;
    let Value::Object(projected_object) = projected.value() else {
        unreachable!("metadata object")
    };
    let mut replacement = source.clone();
    crate::execution::object::replacement::replace_dynamic_definition(
        runtime,
        &mut replacement,
        definition,
        |message, error| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:dynamicprops:Allocation",
                message,
                error,
            )
        },
        |message| {
            dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                message,
            )
        },
    )?;
    if let Some(owner) = &owner {
        owner
            .check(runtime.expect("managed owner context"))
            .map_err(|source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    runtime,
                    "RunMat:dynamicprops:Allocation",
                    "dynamic metadata publication was cancelled",
                    source,
                )
            })?;
    }
    runmat_gc::gc_with_value_mut(&target.target, |target| -> BuiltinResult<()> {
        let Value::Object(current) = target else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                "dynamic property target changed during preparation",
            ));
        };
        if !current.shares_generation_with(&source) {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                "dynamic property target changed during preparation",
            ));
        }
        *current = replacement;
        Ok(())
    })
    .map_err(|source| {
        dynamic_error_with_source(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property target is invalid",
            source,
        )
    })??;
    *metadata = projected_object.clone();
    Ok(true)
}
pub fn object_dynamic_property_names(obj: &ObjectInstance) -> Vec<String> {
    let mut names = obj
        .dynamic_property_names()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    names.sort();
    names
}

pub fn dynamic_property_metadata_struct(def: &DynamicPropertyDef) -> Value {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    fields.insert("Name", Value::string(def.name.clone()));
    fields.insert(
        "DefiningClass",
        Value::ClassRef(def.defining_class.clone().into()),
    );
    fields.insert("GetAccess", access_value(&def.get_access));
    fields.insert("SetAccess", access_value(&def.set_access));
    fields.insert("Dependent", Value::Bool(def.dependent));
    fields.insert("Hidden", Value::Bool(def.hidden));
    fields.insert("Transient", Value::Bool(def.transient));
    fields.insert("NonCopyable", Value::Bool(def.non_copyable));
    fields.insert("AbortSet", Value::Bool(def.abort_set));
    fields.insert("SetObservable", Value::Bool(def.set_observable));
    fields.insert("GetObservable", Value::Bool(def.get_observable));
    fields.insert("Description", Value::string(def.description.clone()));
    Value::Struct(fields.finish())
}
