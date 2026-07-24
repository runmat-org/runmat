//! MATLAB-compatible dynamic property support for handle objects.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    DynamicPropertyDef, HandleRef, ObjectInstance, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use std::collections::HashMap;

pub const DYNAMICPROPS_CLASS: &str = "dynamicprops";
pub const DYNAMIC_PROPERTY_CLASS: &str = "matlab.metadata.DynamicProperty";
pub const STATIC_PROPERTY_METADATA_CLASS: &str = "matlab.metadata.Property";

const TARGET_FIELD: &str = "__runmat_dynamic_property_target__";
const VALID_FIELD: &str = "__runmat_dynamic_property_valid__";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::dynamicprops")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "addprop/findprop",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dynamic property metadata is host-side object state; GPU provider kernels do not participate.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::introspection::dynamicprops"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "addprop/findprop",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Dynamic property mutation is a host-side object metadata side effect.",
};

const ADDPROP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dynamic property metadata handle.",
}];

const ADDPROP_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "object",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar handle object that receives a per-instance dynamic property.",
    },
    BuiltinParamDescriptor {
        name: "property_name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the dynamic property to add.",
    },
];

const FINDPROP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Property metadata object, or an empty double when no property exists.",
}];

const FINDPROP_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "object",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object or handle object to inspect.",
    },
    BuiltinParamDescriptor {
        name: "property_name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Property name to locate.",
    },
];

const DYNAMIC_DELETE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prop",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dynamic property metadata handle to delete.",
}];

const DYNAMICPROPS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "prop = addprop(object, property_name)",
    inputs: &ADDPROP_INPUTS,
    outputs: &ADDPROP_OUTPUT,
}];

const FINDPROP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "prop = findprop(object, property_name)",
    inputs: &FINDPROP_INPUTS,
    outputs: &FINDPROP_OUTPUT,
}];

const DYNAMIC_DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "delete(prop)",
    inputs: &DYNAMIC_DELETE_INPUTS,
    outputs: &[],
}];

const DYNAMIC_ERROR_INVALID_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_TARGET",
    identifier: Some("RunMat:dynamicprops:InvalidTarget"),
    when: "The target is not a valid handle object.",
    message: "dynamicprops: target must be a valid handle object",
};
const DYNAMIC_ERROR_INVALID_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_NAME",
    identifier: Some("RunMat:dynamicprops:InvalidName"),
    when: "The property name is empty or not a valid identifier.",
    message: "dynamicprops: invalid property name",
};
const DYNAMIC_ERROR_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.DUPLICATE_PROPERTY",
    identifier: Some("RunMat:dynamicprops:DuplicateProperty"),
    when: "A class-defined or dynamic property already exists with the requested name.",
    message: "dynamicprops: property already exists",
};
const DYNAMIC_ERROR_INVALID_METADATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.INVALID_METADATA",
    identifier: Some("RunMat:dynamicprops:InvalidMetadata"),
    when: "A dynamic property metadata handle is invalid or malformed.",
    message: "dynamicprops: invalid dynamic property metadata",
};
const DYNAMIC_ERROR_UNSUPPORTED_METADATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DYNAMICPROPS.UNSUPPORTED_METADATA_PROPERTY",
    identifier: Some("RunMat:dynamicprops:UnsupportedMetadataProperty"),
    when: "A metadata property assignment uses an unsupported property name or value.",
    message: "dynamicprops: unsupported dynamic property metadata assignment",
};

const DYNAMIC_ERRORS: [BuiltinErrorDescriptor; 5] = [
    DYNAMIC_ERROR_INVALID_TARGET,
    DYNAMIC_ERROR_INVALID_NAME,
    DYNAMIC_ERROR_DUPLICATE,
    DYNAMIC_ERROR_INVALID_METADATA,
    DYNAMIC_ERROR_UNSUPPORTED_METADATA,
];

pub const ADDPROP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DYNAMICPROPS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_ERRORS,
};

pub const FINDPROP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FINDPROP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_ERRORS,
};

pub const DYNAMIC_PROPERTY_DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DYNAMIC_DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DYNAMIC_ERRORS,
};

fn dynamic_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub fn empty_metadata_result() -> Value {
    Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("empty tensor"))
}

fn validate_property_name(name: &str) -> BuiltinResult<()> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_NAME,
            "addprop: property name must not be empty",
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_NAME,
            format!("addprop: invalid property name '{name}'"),
        ));
    }
    if chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric())) {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_NAME,
            format!("addprop: invalid property name '{name}'"),
        ));
    }
    Ok(())
}

fn empty_default_value() -> Value {
    empty_metadata_result()
}

fn metadata_string(value: &Value, field: &str) -> BuiltinResult<String> {
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

fn metadata_bool(value: &Value, field: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(number) if *number == 0.0 || *number == 1.0 => Ok(*number != 0.0),
        Value::Int(int) => {
            let value = int.to_i64();
            if value == 0 || value == 1 {
                Ok(value != 0)
            } else {
                Err(dynamic_error(
                    DYNAMIC_PROPERTY_CLASS,
                    &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
                    format!("dynamic property metadata field '{field}' requires logical scalar"),
                ))
            }
        }
        _ => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' requires logical scalar"),
        )),
    }
}

fn metadata_access(value: &Value, field: &str) -> BuiltinResult<Access> {
    match metadata_string(value, field)?.to_ascii_lowercase().as_str() {
        "public" => Ok(Access::Public),
        "private" => Ok(Access::Private),
        "protected" => Ok(Access::Protected),
        other => Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("dynamic property metadata field '{field}' has unsupported access '{other}'"),
        )),
    }
}

fn access_value(access: &Access) -> Value {
    let text = match access {
        Access::Public => "public",
        Access::Private => "private",
        Access::Protected => "protected",
    };
    Value::String(text.to_string())
}

fn dynamic_def_to_metadata_object(
    def: &DynamicPropertyDef,
    target: Option<Value>,
    class_name: &str,
) -> ObjectInstance {
    let mut object = ObjectInstance::new(class_name.to_string());
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

fn dynamic_def_from_class_property(
    class_name: &str,
    prop: &runmat_builtins::PropertyDef,
) -> DynamicPropertyDef {
    let mut def = DynamicPropertyDef::new(prop.name.clone(), class_name.to_string());
    def.get_access = prop.get_access.clone();
    def.set_access = prop.set_access.clone();
    def.dependent = prop.is_dependent;
    def
}

fn allocate_dynamic_property_handle(
    def: &DynamicPropertyDef,
    target: Value,
) -> BuiltinResult<HandleRef> {
    let metadata = dynamic_def_to_metadata_object(def, Some(target), DYNAMIC_PROPERTY_CLASS);
    let handle = runmat_gc::gc_allocate(Value::Object(metadata))
        .map_err(|err| format!("addprop: failed to allocate metadata handle: {err}"))?;
    Ok(HandleRef {
        class_name: DYNAMIC_PROPERTY_CLASS.to_string(),
        target: handle,
        valid: true,
    })
}

fn dynamic_property_handle_from_gc(target: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: DYNAMIC_PROPERTY_CLASS.to_string(),
        target,
        valid: true,
    })
}

fn require_dynamicprops_target(class_name: &str) -> BuiltinResult<()> {
    if runmat_builtins::is_class_or_subclass(class_name, DYNAMICPROPS_CLASS) {
        Ok(())
    } else {
        Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("addprop: class '{class_name}' is not a dynamicprops subclass"),
        ))
    }
}

fn metadata_target_and_name(metadata: &ObjectInstance) -> BuiltinResult<(HandleRef, String)> {
    if metadata.class_name != DYNAMIC_PROPERTY_CLASS {
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
    let Some(def) = obj.dynamic_property(name) else {
        return Ok(false);
    };
    if def.set_access == Access::Private {
        return Err(dynamic_error(
            "setfield",
            &DYNAMIC_ERROR_UNSUPPORTED_METADATA,
            format!("Property '{name}' is private"),
        ));
    }
    obj.properties.insert(name.to_string(), value);
    Ok(true)
}

pub fn dynamic_property_read(obj: &ObjectInstance, name: &str) -> BuiltinResult<Option<Value>> {
    let Some((def, value)) = dynamic_property_get(obj, name) else {
        return Ok(None);
    };
    if def.get_access == Access::Private {
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
    if metadata.class_name != DYNAMIC_PROPERTY_CLASS {
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
    .map_err(|err| {
        dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            format!("dynamic property target is invalid: {err}"),
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

fn remove_dynamic_property(target: &HandleRef, name: &str) -> BuiltinResult<()> {
    if !crate::is_handle_valid(target) {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            "dynamic property target handle is invalid",
        ));
    }
    runmat_gc::gc_with_value_mut(&target.target, |target_value| -> BuiltinResult<()> {
        let Value::Object(target_obj) = target_value else {
            return Err(dynamic_error(
                DYNAMIC_PROPERTY_CLASS,
                &DYNAMIC_ERROR_INVALID_TARGET,
                "dynamic property target is not an object",
            ));
        };
        target_obj.remove_dynamic_property(name);
        target_obj.properties.remove(name);
        Ok(())
    })
    .map_err(|err| {
        dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("dynamic property target is invalid: {err}"),
        )
    })?
}

#[runtime_builtin(
    name = "addprop",
    category = "introspection",
    summary = "Add a per-instance dynamic property to a handle object.",
    keywords = "dynamicprops,addprop,dynamic property,handle object,meta.DynamicProperty",
    sink = true,
    descriptor(crate::builtins::introspection::dynamicprops::ADDPROP_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamicprops"
)]
async fn addprop_builtin(target: Value, property_name: String) -> BuiltinResult<Value> {
    validate_property_name(&property_name)?;
    let Value::HandleObject(handle) = target.clone() else {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: target must be a handle object",
        ));
    };
    if !crate::is_handle_valid(&handle) {
        return Err(dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            "addprop: target handle is invalid",
        ));
    }
    let mut inserted = None;
    runmat_gc::gc_with_value_mut(&handle.target, |target_value| -> BuiltinResult<()> {
        let Value::Object(obj) = target_value else {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_INVALID_TARGET,
                "addprop: handle target is not an object",
            ));
        };
        require_dynamicprops_target(&obj.class_name)?;
        if runmat_builtins::lookup_property(&obj.class_name, &property_name).is_some()
            || obj.has_dynamic_property(&property_name)
        {
            return Err(dynamic_error(
                "addprop",
                &DYNAMIC_ERROR_DUPLICATE,
                format!("addprop: property '{property_name}' already exists"),
            ));
        }
        let def = DynamicPropertyDef::new(property_name.clone(), obj.class_name.clone());
        obj.insert_dynamic_property(property_name.clone(), def.clone());
        obj.properties
            .entry(property_name.clone())
            .or_insert_with(empty_default_value);
        inserted = Some(def);
        Ok(())
    })
    .map_err(|err| {
        dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("addprop: invalid handle target: {err}"),
        )
    })??;
    let def = inserted.expect("dynamic property inserted before successful return");
    let metadata_handle = allocate_dynamic_property_handle(&def, target)?;
    runmat_gc::gc_with_value_mut(&handle.target, |target_value| {
        if let Value::Object(obj) = target_value {
            if let Some(def) = obj.dynamic_property_mut(&property_name) {
                def.metadata_handle = Some(metadata_handle.target);
            }
        }
    })
    .map_err(|err| {
        dynamic_error(
            "addprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("addprop: invalid handle target: {err}"),
        )
    })?;
    Ok(Value::HandleObject(metadata_handle))
}

#[runtime_builtin(
    name = "findprop",
    category = "introspection",
    summary = "Find class-defined or dynamic property metadata on an object.",
    keywords = "dynamicprops,findprop,property metadata,meta.property,meta.DynamicProperty",
    descriptor(crate::builtins::introspection::dynamicprops::FINDPROP_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamicprops"
)]
fn findprop_builtin(target: Value, property_name: String) -> BuiltinResult<Value> {
    validate_property_name(&property_name)?;
    match target {
        Value::HandleObject(handle) => {
            if !crate::is_handle_valid(&handle) {
                return Err(dynamic_error(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    "findprop: target handle is invalid",
                ));
            }
            let target_value = runmat_gc::gc_clone_value(&handle.target).map_err(|err| {
                dynamic_error(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    format!("findprop: invalid handle target: {err}"),
                )
            })?;
            let Value::Object(obj) = target_value else {
                return Err(dynamic_error(
                    "findprop",
                    &DYNAMIC_ERROR_INVALID_TARGET,
                    "findprop: handle target is not an object",
                ));
            };
            if let Some(def) = obj.dynamic_property(&property_name) {
                if let Some(metadata_handle) = def.metadata_handle {
                    return Ok(dynamic_property_handle_from_gc(metadata_handle));
                }
                return Ok(Value::HandleObject(allocate_dynamic_property_handle(
                    def,
                    Value::HandleObject(handle.clone()),
                )?));
            }
            if let Some((prop, owner)) =
                runmat_builtins::lookup_property(&obj.class_name, &property_name)
            {
                let def = dynamic_def_from_class_property(&owner, &prop);
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                )));
            }
            Ok(empty_metadata_result())
        }
        Value::Object(obj) => {
            if let Some(def) = obj.dynamic_property(&property_name) {
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    def,
                    None,
                    DYNAMIC_PROPERTY_CLASS,
                )));
            }
            if let Some((prop, owner)) =
                runmat_builtins::lookup_property(&obj.class_name, &property_name)
            {
                let def = dynamic_def_from_class_property(&owner, &prop);
                return Ok(Value::Object(dynamic_def_to_metadata_object(
                    &def,
                    None,
                    STATIC_PROPERTY_METADATA_CLASS,
                )));
            }
            Ok(empty_metadata_result())
        }
        other => Err(dynamic_error(
            "findprop",
            &DYNAMIC_ERROR_INVALID_TARGET,
            format!("findprop: target must be object or handle object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "matlab.metadata.DynamicProperty.delete",
    category = "introspection",
    summary = "Delete a dynamic property metadata handle and remove its target property.",
    keywords = "dynamicprops,delete,dynamic property,meta.DynamicProperty",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::dynamicprops::DYNAMIC_PROPERTY_DELETE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dynamicprops"
)]
async fn dynamic_property_delete_builtin(prop: Value) -> BuiltinResult<Value> {
    let Value::HandleObject(prop_handle) = prop else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property delete requires a metadata handle",
        ));
    };
    if prop_handle.class_name != DYNAMIC_PROPERTY_CLASS {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property delete requires matlab.metadata.DynamicProperty",
        ));
    }
    let metadata = runmat_gc::gc_clone_value(&prop_handle.target).map_err(|err| {
        dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            format!("dynamic property metadata handle is invalid: {err}"),
        )
    })?;
    let Value::Object(metadata) = metadata else {
        return Err(dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            "dynamic property metadata target is not an object",
        ));
    };
    let (target, name) = metadata_target_and_name(&metadata)?;
    remove_dynamic_property(&target, &name)?;
    runmat_gc::gc_with_value_mut(&prop_handle.target, |target_value| {
        if let Value::Object(metadata) = target_value {
            metadata
                .properties
                .insert(VALID_FIELD.to_string(), Value::Bool(false));
            metadata.properties.remove(TARGET_FIELD);
        }
    })
    .map_err(|err| {
        dynamic_error(
            DYNAMIC_PROPERTY_CLASS,
            &DYNAMIC_ERROR_INVALID_METADATA,
            format!("dynamic property metadata handle is invalid: {err}"),
        )
    })?;
    crate::set_handle_valid(&prop_handle, false);
    Ok(Value::Num(0.0))
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn target_handle(class_name: &str) -> (Value, runmat_gc::ExplicitRoot) {
        let obj = ObjectInstance::new(class_name.to_string());
        let root = runmat_gc::gc_allocate_rooted(Value::Object(obj)).expect("gc");
        let value = Value::HandleObject(HandleRef {
            class_name: class_name.to_string(),
            target: root.handle(),
            valid: true,
        });
        (value, root)
    }

    fn dynamic_target_handle(class_name: &str) -> (Value, runmat_gc::ExplicitRoot) {
        runmat_builtins::register_class(runmat_builtins::ClassDef {
            name: class_name.to_string(),
            parent: Some(DYNAMICPROPS_CLASS.to_string()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        target_handle(class_name)
    }

    #[test]
    fn addprop_adds_dynamic_property_and_returns_metadata_handle() {
        {
            let (target, _target_root) = dynamic_target_handle("DynTarget");
            let prop =
                block_on(addprop_builtin(target.clone(), "gain".to_string())).expect("addprop");
            assert!(matches!(prop, Value::HandleObject(_)));
            let found = findprop_builtin(target.clone(), "gain".to_string()).expect("findprop");
            assert!(matches!(found, Value::HandleObject(_)));
            let Value::HandleObject(handle) = target else {
                panic!("target handle");
            };
            let target_obj = runmat_gc::gc_clone_value(&handle.target).expect("target");
            let Value::Object(obj) = target_obj else {
                panic!("target object");
            };
            assert!(obj.has_dynamic_property("gain"));
            assert!(obj.properties.contains_key("gain"));
        }
    }

    #[test]
    fn dynamic_property_delete_removes_target_property() {
        {
            let (target, _target_root) = dynamic_target_handle("DynTarget");
            let prop =
                block_on(addprop_builtin(target.clone(), "gain".to_string())).expect("addprop");
            block_on(dynamic_property_delete_builtin(prop)).expect("delete");
            let Value::HandleObject(handle) = target else {
                panic!("target handle");
            };
            let target_obj = runmat_gc::gc_clone_value(&handle.target).expect("target");
            let Value::Object(obj) = target_obj else {
                panic!("target object");
            };
            assert!(!obj.has_dynamic_property("gain"));
            assert!(!obj.properties.contains_key("gain"));
        }
    }

    #[test]
    fn dynamic_property_supports_value_access_and_metadata_mutation() {
        {
            let (target, _target_root) = dynamic_target_handle("DynTarget");
            let prop =
                block_on(addprop_builtin(target.clone(), "gain".to_string())).expect("addprop");

            block_on(crate::call_builtin_async(
                "setfield",
                &[
                    target.clone(),
                    Value::String("gain".to_string()),
                    Value::Num(7.0),
                ],
            ))
            .expect("set dynamic property");
            let value = block_on(crate::call_builtin_async(
                "getfield",
                &[target.clone(), Value::String("gain".to_string())],
            ))
            .expect("get dynamic property");
            assert_eq!(value, Value::Num(7.0));

            block_on(crate::call_builtin_async(
                "setfield",
                &[
                    prop.clone(),
                    Value::String("Hidden".to_string()),
                    Value::Bool(true),
                ],
            ))
            .expect("set metadata");

            let Value::HandleObject(target_handle) = target else {
                panic!("target handle");
            };
            let target_obj = runmat_gc::gc_clone_value(&target_handle.target).expect("target");
            let Value::Object(obj) = target_obj else {
                panic!("target object");
            };
            assert!(
                obj.dynamic_property("gain")
                    .expect("dynamic property")
                    .hidden
            );
        }
    }

    #[test]
    fn addprop_rejects_handle_classes_without_dynamicprops_parent() {
        {
            runmat_builtins::register_class(runmat_builtins::ClassDef {
                name: "PlainHandleForDynamicProps".to_string(),
                parent: Some("handle".to_string()),
                properties: HashMap::new(),
                methods: HashMap::new(),
            });
            let (target, _target_root) = target_handle("PlainHandleForDynamicProps");
            let err = block_on(addprop_builtin(target, "gain".to_string()))
                .expect_err("plain handles cannot add dynamic properties");
            assert!(
                err.to_string().contains("not a dynamicprops subclass"),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn findprop_aliases_share_delete_invalidation() {
        {
            let (target, _target_root) = dynamic_target_handle("DynAliasTarget");
            let first =
                block_on(addprop_builtin(target.clone(), "gain".to_string())).expect("addprop");
            let second = findprop_builtin(target.clone(), "gain".to_string()).expect("findprop");
            let (Value::HandleObject(first_handle), Value::HandleObject(second_handle)) =
                (&first, &second)
            else {
                panic!("metadata handles");
            };
            assert_eq!(first_handle.target, second_handle.target);

            block_on(dynamic_property_delete_builtin(second)).expect("delete alias");
            assert!(!crate::is_handle_valid(first_handle));
            let Value::HandleObject(metadata) = first else {
                panic!("metadata handle");
            };
            let metadata_value = runmat_gc::gc_clone_value(&metadata.target).expect("metadata");
            let Value::Object(metadata_obj) = metadata_value else {
                panic!("metadata object");
            };
            assert!(!metadata_obj.properties.contains_key(TARGET_FIELD));
        }
    }
}
