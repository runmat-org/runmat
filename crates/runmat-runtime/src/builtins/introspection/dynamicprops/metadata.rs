use super::{DYNAMICPROPS_CLASS, DYNAMIC_PROPERTY_CLASS, TARGET_FIELD, VALID_FIELD};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, DYNAMIC_ERROR_INVALID_METADATA, DYNAMIC_ERROR_INVALID_NAME,
    DYNAMIC_ERROR_INVALID_TARGET, DYNAMIC_ERROR_UNSUPPORTED_METADATA,
};
use runmat_types::MemberAccess;
use runmat_value::{DynamicPropertyDef, HandleRef, ObjectInstance, StructValue, Tensor, Value};

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

pub(super) fn metadata_string(value: &Value, field: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
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

pub(super) fn metadata_access(value: &Value, field: &str) -> BuiltinResult<MemberAccess> {
    match metadata_string(value, field)?.to_ascii_lowercase().as_str() {
        "public" => Ok(MemberAccess::Public),
        "private" => Ok(MemberAccess::Private),
        "protected" => Ok(MemberAccess::Protected),
        other => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' has unsupported access '{other}'"),
        )),
    }
}

pub(super) fn access_value(access: &MemberAccess) -> Value {
    let text = match access {
        MemberAccess::Public => "public",
        MemberAccess::Private => "private",
        MemberAccess::Protected => "protected",
    };
    Value::String(text.to_string())
}

pub(super) fn dynamic_def_to_metadata_object(
    def: &DynamicPropertyDef,
    target: Option<Value>,
    class_name: runmat_types::StaticClassIdentity,
) -> ObjectInstance {
    let mut object = ObjectInstance::new(class_name);
    object
        .properties
        .insert("Name".to_string(), Value::String(def.name.clone()));
    object.properties.insert(
        "DefiningClass".to_string(),
        Value::ClassRef(def.defining_class.clone()),
    );
    object
        .properties
        .insert("GetAccess".to_string(), access_value(&def.get_access));
    object
        .properties
        .insert("SetAccess".to_string(), access_value(&def.set_access));
    object
        .properties
        .insert("Dependent".to_string(), Value::Bool(def.dependent));
    object
        .properties
        .insert("Hidden".to_string(), Value::Bool(def.hidden));
    object
        .properties
        .insert("Transient".to_string(), Value::Bool(def.transient));
    object
        .properties
        .insert("NonCopyable".to_string(), Value::Bool(def.non_copyable));
    object
        .properties
        .insert("AbortSet".to_string(), Value::Bool(def.abort_set));
    object
        .properties
        .insert("SetObservable".to_string(), Value::Bool(def.set_observable));
    object
        .properties
        .insert("GetObservable".to_string(), Value::Bool(def.get_observable));
    object.properties.insert(
        "Description".to_string(),
        Value::String(def.description.clone()),
    );
    object
        .properties
        .insert(VALID_FIELD.to_string(), Value::Bool(true));
    if let Some(target) = target {
        object.properties.insert(TARGET_FIELD.to_string(), target);
    }
    object
}

pub(super) fn dynamic_def_from_class_property(
    class_name: &runmat_types::ClassIdentity,
    name: &runmat_types::MemberName,
    prop: &runmat_value::SelectedReceiverPropertyDeclaration,
) -> DynamicPropertyDef {
    let mut def = DynamicPropertyDef::new(name.to_string(), class_name.clone());
    def.get_access = prop.get_access;
    def.set_access = prop.set_access;
    def.dependent = prop.is_dependent;
    def
}

pub(super) fn allocate_dynamic_property_handle(
    def: &DynamicPropertyDef,
    target: Value,
) -> BuiltinResult<HandleRef> {
    let metadata = dynamic_def_to_metadata_object(def, Some(target), DYNAMIC_PROPERTY_CLASS);
    let handle = runmat_gc::gc_allocate(Value::Object(metadata)).map_err(|source| {
        dynamic_error_with_source(
            "addprop",
            &runmat_builtins::ADDPROP_ERROR_ALLOCATION,
            "addprop: failed to allocate metadata handle",
            source,
        )
    })?;
    Ok(HandleRef {
        class_name: DYNAMIC_PROPERTY_CLASS.into(),
        target: handle,
        valid: true,
    })
}

pub(super) fn dynamic_property_handle_from_gc(target: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: DYNAMIC_PROPERTY_CLASS.into(),
        target,
        valid: true,
    })
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
    metadata: &ObjectInstance,
) -> BuiltinResult<(HandleRef, String)> {
    if !metadata.class_name.is(DYNAMIC_PROPERTY_CLASS) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property operation requires matlab.metadata.DynamicProperty",
        ));
    }
    if !matches!(
        metadata.properties.get(VALID_FIELD),
        Some(Value::Bool(true))
    ) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata handle is invalid",
        ));
    }
    let Some(Value::HandleObject(target)) = metadata.properties.get(TARGET_FIELD).cloned() else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata is missing its target object",
        ));
    };
    let name = metadata_string(
        metadata.properties.get("Name").ok_or_else(|| {
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
        .properties
        .get(name)
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
    obj.properties.insert(name.to_string(), value);
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
        MemberAccess::Private => caller == Some(&def.defining_class),
        MemberAccess::Protected => caller.is_some_and(|caller| {
            crate::class_registry::is_class_or_subclass(caller, &def.defining_class)
        }),
    })
}

pub fn dynamic_property_read(obj: &ObjectInstance, name: &str) -> BuiltinResult<Option<Value>> {
    let Some((def, value)) = dynamic_property_get(obj, name) else {
        return Ok(None);
    };
    if let Some(projection) = crate::class_registry::checked_receiver_projection(obj)? {
        if !projection.receiver_member_accessible(
            &def.defining_class,
            def.get_access,
            crate::dispatcher::class_access_context().as_ref(),
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
    metadata: &mut ObjectInstance,
    field: &str,
    value: Value,
) -> BuiltinResult<bool> {
    if !metadata.class_name.is(DYNAMIC_PROPERTY_CLASS) {
        return Ok(false);
    }
    if field == "Name" || field == "DefiningClass" {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' is read-only"),
        ));
    }
    let (target, name) = metadata_target_and_name(metadata)?;
    let update = |def: &mut DynamicPropertyDef| -> BuiltinResult<()> {
        match field {
            "GetAccess" => def.get_access = metadata_access(&value, field)?,
            "SetAccess" => def.set_access = metadata_access(&value, field)?,
            "Dependent" => def.dependent = metadata_bool(&value, field)?,
            "Hidden" => def.hidden = metadata_bool(&value, field)?,
            "Transient" => def.transient = metadata_bool(&value, field)?,
            "NonCopyable" => def.non_copyable = metadata_bool(&value, field)?,
            "AbortSet" => def.abort_set = metadata_bool(&value, field)?,
            "SetObservable" => def.set_observable = metadata_bool(&value, field)?,
            "GetObservable" => def.get_observable = metadata_bool(&value, field)?,
            "Description" => def.description = metadata_string(&value, field)?,
            _ => {
                return Err(dynamic_error(
                    DYNAMIC_PROPERTY_CLASS,
                    &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
                    format!("unsupported dynamic property metadata field '{field}'"),
                ))
            }
        }
        Ok(())
    };
    let mut updated_def = None;
    runmat_gc::gc_with_value_mut(&target.target, |target_value| -> BuiltinResult<()> {
        let Value::Object(target_obj) = target_value else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                "dynamic property target is not an object",
            ));
        };
        let Some(def) = target_obj.dynamic_property_mut(&name) else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_METADATA,
                format!("dynamic property '{name}' no longer exists"),
            ));
        };
        update(def)?;
        updated_def = Some(def.clone());
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
    if let Some(def) = updated_def {
        *metadata = dynamic_def_to_metadata_object(
            &def,
            Some(Value::HandleObject(target)),
            DYNAMIC_PROPERTY_CLASS,
        );
    }
    Ok(true)
}

pub fn object_dynamic_property_names(obj: &ObjectInstance) -> Vec<String> {
    let mut names = obj.dynamic_property_names();
    names.sort();
    names
}

pub fn dynamic_property_metadata_struct(def: &DynamicPropertyDef) -> Value {
    let mut fields = StructValue::new();
    fields.insert("Name", Value::String(def.name.clone()));
    fields.insert("DefiningClass", Value::ClassRef(def.defining_class.clone()));
    fields.insert("GetAccess", access_value(&def.get_access));
    fields.insert("SetAccess", access_value(&def.set_access));
    fields.insert("Dependent", Value::Bool(def.dependent));
    fields.insert("Hidden", Value::Bool(def.hidden));
    fields.insert("Transient", Value::Bool(def.transient));
    fields.insert("NonCopyable", Value::Bool(def.non_copyable));
    fields.insert("AbortSet", Value::Bool(def.abort_set));
    fields.insert("SetObservable", Value::Bool(def.set_observable));
    fields.insert("GetObservable", Value::Bool(def.get_observable));
    fields.insert("Description", Value::String(def.description.clone()));
    Value::Struct(fields)
}
