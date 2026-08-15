//! Object save/load customization hooks.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, ObjectInstance, StructValue, Value,
};
use runmat_macros::runtime_builtin;

pub(crate) const SAVEOBJ_METHOD: &str = "saveobj";
pub(crate) const LOADOBJ_METHOD: &str = "loadobj";

const SERIALIZED_CLASS_FIELD: &str = "__runmat_serialized_object_class__";
const SERIALIZED_PAYLOAD_FIELD: &str = "__runmat_serialized_object_payload__";
const SERIALIZED_KIND_FIELD: &str = "__runmat_serialized_object_kind__";
const SERIALIZED_HAD_SAVEOBJ_FIELD: &str = "__runmat_serialized_object_had_saveobj__";
const SERIALIZED_KIND_VALUE: &str = "value";
const SERIALIZED_KIND_HANDLE: &str = "handle";
const MAX_SERIALIZATION_DEPTH: usize = 32;

const SAVEOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized object representation.",
}];

const SAVEOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object to serialize.",
}];

const SAVEOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "b = saveobj(a)",
    inputs: &SAVEOBJ_INPUTS,
    outputs: &SAVEOBJ_OUTPUT,
}];

const LOADOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Deserialized object.",
}];

const LOADOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized object representation.",
}];

const LOADOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "b = loadobj(a)",
    inputs: &LOADOBJ_INPUTS,
    outputs: &LOADOBJ_OUTPUT,
}];

const SERIALIZATION_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:ObjectSerialization:InvalidArgument"),
    when: "The input is not an object or serialized object envelope.",
    message: "object serialization: invalid argument",
};

const SERIALIZATION_ERROR_DISPATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.DISPATCH",
    identifier: Some("RunMat:ObjectSerialization:Dispatch"),
    when: "A saveobj or loadobj method cannot be resolved or executed.",
    message: "object serialization: method dispatch failed",
};

const SERIALIZATION_ERROR_RECURSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OBJECT_SERIALIZATION.RECURSION",
    identifier: Some("RunMat:ObjectSerialization:RecursionLimit"),
    when: "Object serialization nesting exceeds the supported recursion limit.",
    message: "object serialization: recursion limit exceeded",
};

const SERIALIZATION_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SERIALIZATION_ERROR_ARGUMENT,
    SERIALIZATION_ERROR_DISPATCH,
    SERIALIZATION_ERROR_RECURSION,
];

pub const SAVEOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVEOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SERIALIZATION_ERRORS,
};

pub const SAVEOBJ_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "saveobj dispatches only on value or handle objects; direct typed-integer input is invalid. Integer-valued object properties remain opaque nested payloads and are copied through authoritative storage without becoming a direct integer call form.",
};

pub const LOADOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOADOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SERIALIZATION_ERRORS,
};

const LOADOBJ_PASSTHROUGH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "loadobj-plain-payload-passthrough",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "loadobj passthrough for a plain non-object, non-structure payload is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LoadobjPlainPayloadPassthroughExtension"),
};

pub const LOADOBJ_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LOADOBJ_PASSTHROUGH_EXTENSION];

const LOADOBJ_INTEGER_PAYLOAD_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "a.integer_payload",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An object or structure supplied to loadobj may contain fields of every integer class; nested payloads retain exact class, shape, and value through recursive restoration and class loadobj dispatch.",
    }];

const LOADOBJ_DIRECT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "integer_a",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A direct integer payload is outside the documented object-or-structure surface and is admitted only in RunMat mode as an exact passthrough.",
    }];

pub const LOADOBJ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "b = loadobj(a_with_integer_payload)",
        inputs: &LOADOBJ_INTEGER_PAYLOAD_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer fields are opaque serialized payloads rather than numeric controls and are never materialized as floating point by the default restoration path.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = loadobj(integer_a)",
        inputs: &LOADOBJ_DIRECT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat mode preserves a direct integer scalar or array without conversion; MATLAB-compatible modes reject this undocumented plain-payload form before recursive restoration.",
    },
];

fn serialization_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(builtin, descriptor, detail.into())
}

fn object_class_name(value: &Value) -> Option<String> {
    crate::object_receiver_class_name(value)
}

async fn dispatch_registered_method(
    class_name: &str,
    method_name: &str,
    args: Vec<Value>,
    builtin: &'static str,
) -> crate::BuiltinResult<Option<Value>> {
    if let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) {
        let owner_member = format!("{owner}.{method_name}");
        let mut candidates = Vec::with_capacity(2);
        if !method.function_name.trim().is_empty() {
            candidates.push(method.function_name);
        }
        if !candidates
            .iter()
            .any(|candidate| candidate == &owner_member)
        {
            candidates.push(owner_member);
        }

        let mut undefined = None;
        for candidate in candidates {
            let (identity, fallback_policy) = crate::callable_identity_for_handle_name(&candidate);
            match crate::dispatch_callable_with_policy(identity, fallback_policy, args.clone(), 1)
                .await
            {
                Ok(value) => return Ok(Some(value)),
                Err(err) if crate::is_undefined_function_error(&err) => undefined = Some(err),
                Err(err) => return Err(err),
            }
        }
        return Err(undefined.unwrap_or_else(|| {
            serialization_error(
                builtin,
                &SERIALIZATION_ERROR_DISPATCH,
                format!("{class_name}.{method_name} did not resolve to a callable"),
            )
        }));
    }

    if args
        .first()
        .is_some_and(|arg| matches!(arg, Value::Object(_) | Value::HandleObject(_)))
    {
        match crate::dispatch_object_external_member(class_name.to_string(), method_name, args, 1)
            .await
        {
            Ok(value) => Ok(Some(value)),
            Err(err) if crate::is_undefined_function_error(&err) => Ok(None),
            Err(err) => Err(err),
        }
    } else {
        Ok(None)
    }
}

async fn call_saveobj_if_available(value: Value) -> crate::BuiltinResult<Option<Value>> {
    let Some(class_name) = object_class_name(&value) else {
        return Ok(None);
    };
    dispatch_registered_method(&class_name, SAVEOBJ_METHOD, vec![value], SAVEOBJ_METHOD).await
}

async fn call_loadobj_if_available(
    class_name: &str,
    payload: Value,
) -> crate::BuiltinResult<Option<Value>> {
    dispatch_registered_method(class_name, LOADOBJ_METHOD, vec![payload], LOADOBJ_METHOD).await
}

fn object_properties_payload(value: &Value) -> crate::BuiltinResult<StructValue> {
    match value {
        Value::Object(object) => {
            let mut payload = StructValue::new();
            for (name, property) in &object.properties {
                payload.insert(name.clone(), property.clone());
            }
            Ok(payload)
        }
        Value::HandleObject(handle) => runmat_gc::gc_with_value(&handle.target, |target| {
            if let Value::Object(object) = target {
                let mut payload = StructValue::new();
                for (name, property) in &object.properties {
                    payload.insert(name.clone(), property.clone());
                }
                Ok(payload)
            } else {
                Err(serialization_error(
                    SAVEOBJ_METHOD,
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "handle target is not an object",
                ))
            }
        })
        .map_err(|err| {
            serialization_error(
                SAVEOBJ_METHOD,
                &SERIALIZATION_ERROR_ARGUMENT,
                format!("failed to read handle target: {err}"),
            )
        })?,
        other => Err(serialization_error(
            SAVEOBJ_METHOD,
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

fn serialized_object_envelope(
    class_name: String,
    kind: &'static str,
    had_saveobj: bool,
    payload: Value,
) -> Value {
    let mut st = StructValue::new();
    st.insert(SERIALIZED_CLASS_FIELD, Value::String(class_name));
    st.insert(SERIALIZED_KIND_FIELD, Value::String(kind.to_string()));
    st.insert(SERIALIZED_HAD_SAVEOBJ_FIELD, Value::Bool(had_saveobj));
    st.insert(SERIALIZED_PAYLOAD_FIELD, payload);
    Value::Struct(st)
}

fn serialized_envelope(value: &Value) -> Option<(String, Value)> {
    let Value::Struct(st) = value else {
        return None;
    };
    let class_name = st
        .fields
        .get(SERIALIZED_CLASS_FIELD)
        .and_then(|value| String::try_from(value).ok())?;
    let payload = st.fields.get(SERIALIZED_PAYLOAD_FIELD)?.clone();
    if st.fields.get(SERIALIZED_KIND_FIELD).is_none()
        || st.fields.get(SERIALIZED_HAD_SAVEOBJ_FIELD).is_none()
    {
        return None;
    }
    Some((class_name, payload))
}

async fn prepare_value_for_save_depth(value: Value, depth: usize) -> crate::BuiltinResult<Value> {
    if depth > MAX_SERIALIZATION_DEPTH {
        return Err(serialization_error(
            SAVEOBJ_METHOD,
            &SERIALIZATION_ERROR_RECURSION,
            "nested object serialization exceeded the supported depth",
        ));
    }

    match value {
        receiver @ (Value::Object(_) | Value::HandleObject(_)) => {
            let class_name = object_class_name(&receiver).ok_or_else(|| {
                serialization_error(
                    SAVEOBJ_METHOD,
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "object receiver is missing class metadata",
                )
            })?;
            let kind = if matches!(receiver, Value::HandleObject(_)) {
                SERIALIZED_KIND_HANDLE
            } else {
                SERIALIZED_KIND_VALUE
            };
            let (payload, had_saveobj) =
                if let Some(saved) = call_saveobj_if_available(receiver.clone()).await? {
                    (saved, true)
                } else {
                    (Value::Struct(object_properties_payload(&receiver)?), false)
                };
            let payload = Box::pin(prepare_value_for_save_depth(payload, depth + 1)).await?;
            Ok(serialized_object_envelope(
                class_name,
                kind,
                had_saveobj,
                payload,
            ))
        }
        Value::Struct(st) => {
            let mut converted = StructValue::new();
            for (name, field) in st.fields.into_iter() {
                converted.insert(
                    name,
                    Box::pin(prepare_value_for_save_depth(field, depth + 1)).await?,
                );
            }
            Ok(Value::Struct(converted))
        }
        Value::Cell(cell) => {
            let mut converted = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                converted.push(Box::pin(prepare_value_for_save_depth(value, depth + 1)).await?);
            }
            CellArray::new_with_shape(converted, cell.shape)
                .map(Value::Cell)
                .map_err(|err| {
                    serialization_error(
                        SAVEOBJ_METHOD,
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("failed to rebuild serialized cell payload: {err}"),
                    )
                })
        }
        other => Ok(other),
    }
}

pub(crate) async fn prepare_value_for_mat_save(value: Value) -> crate::BuiltinResult<Value> {
    prepare_value_for_save_depth(value, 0).await
}

async fn restore_value_after_load_depth(value: Value, depth: usize) -> crate::BuiltinResult<Value> {
    if depth > MAX_SERIALIZATION_DEPTH {
        return Err(serialization_error(
            LOADOBJ_METHOD,
            &SERIALIZATION_ERROR_RECURSION,
            "nested object deserialization exceeded the supported depth",
        ));
    }

    if let Some((class_name, payload)) = serialized_envelope(&value) {
        let restored_payload = Box::pin(restore_value_after_load_depth(payload, depth + 1)).await?;
        if let Some(restored) =
            call_loadobj_if_available(&class_name, restored_payload.clone()).await?
        {
            return Ok(restored);
        }
        if let Value::Struct(fields) = restored_payload {
            return Ok(Value::Object(ObjectInstance {
                class_name,
                properties: fields.fields.into_iter().collect(),
                dynamic_properties: None,
            }));
        }
        return Ok(restored_payload);
    }

    match value {
        Value::Struct(st) => {
            let mut converted = StructValue::new();
            for (name, field) in st.fields.into_iter() {
                converted.insert(
                    name,
                    Box::pin(restore_value_after_load_depth(field, depth + 1)).await?,
                );
            }
            Ok(Value::Struct(converted))
        }
        Value::Cell(cell) => {
            let mut converted = Vec::with_capacity(cell.data.len());
            for value in cell.data {
                converted.push(Box::pin(restore_value_after_load_depth(value, depth + 1)).await?);
            }
            CellArray::new_with_shape(converted, cell.shape)
                .map(Value::Cell)
                .map_err(|err| {
                    serialization_error(
                        LOADOBJ_METHOD,
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("failed to rebuild deserialized cell payload: {err}"),
                    )
                })
        }
        other => Ok(other),
    }
}

pub(crate) async fn restore_value_from_mat_load(value: Value) -> crate::BuiltinResult<Value> {
    restore_value_after_load_depth(value, 0).await
}

#[runtime_builtin(
    name = "saveobj",
    category = "introspection",
    summary = "Return the serialized representation for an object.",
    keywords = "saveobj,loadobj,object,serialization,mat",
    descriptor(crate::builtins::introspection::object_serialization::SAVEOBJ_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::object_serialization::SAVEOBJ_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn saveobj_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        receiver @ (Value::Object(_) | Value::HandleObject(_)) => {
            if let Some(saved) = call_saveobj_if_available(receiver.clone()).await? {
                Ok(saved)
            } else {
                Ok(Value::Struct(object_properties_payload(&receiver)?))
            }
        }
        other => Err(serialization_error(
            SAVEOBJ_METHOD,
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "loadobj",
    category = "introspection",
    summary = "Restore an object from a serialized representation.",
    keywords = "loadobj,saveobj,object,serialization,mat",
    descriptor(crate::builtins::introspection::object_serialization::LOADOBJ_DESCRIPTOR),
    extensions(crate::builtins::introspection::object_serialization::LOADOBJ_EXTENSIONS),
    integer_capabilities(
        crate::builtins::introspection::object_serialization::LOADOBJ_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn loadobj_builtin(value: Value) -> crate::BuiltinResult<Value> {
    if !matches!(
        value,
        Value::Object(_) | Value::HandleObject(_) | Value::Struct(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOADOBJ_PASSTHROUGH_EXTENSION,
            LOADOBJ_METHOD,
        )?;
    }
    restore_value_from_mat_load(value).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};
    use tempfile::tempdir;

    #[test]
    fn loadobj_preserves_nested_integer_payloads_without_materialization() {
        let mut payload = StructValue::new();
        payload.insert("scalar", Value::Int(IntValue::U64(u64::MAX)));
        payload.insert(
            "array",
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
                    .expect("integer payload"),
            ),
        );
        let restored = block_on(loadobj_builtin(Value::Struct(payload))).expect("loadobj");
        let Value::Struct(restored) = restored else {
            panic!("expected structure payload");
        };
        assert_eq!(
            restored.fields.get("scalar"),
            Some(&Value::Int(IntValue::U64(u64::MAX)))
        );
        let Value::Tensor(array) = restored.fields.get("array").expect("array") else {
            panic!("expected integer array");
        };
        assert_eq!(
            array.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );
    }

    #[test]
    fn loadobj_plain_integer_passthrough_is_declared_and_mode_gated() {
        assert_eq!(LOADOBJ_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(LOADOBJ_EXTENSIONS.len(), 1);
        let value = Value::Int(IntValue::U64(u64::MAX));
        let runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            block_on(loadobj_builtin(value.clone())).expect("RunMat passthrough"),
            value
        );
        drop(runmat);
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(loadobj_builtin(value)).expect_err("strict mode must reject");
        drop(strict);
        assert_eq!(
            error.identifier(),
            LOADOBJ_PASSTHROUGH_EXTENSION.error_identifier
        );
    }

    #[test]
    fn saveobj_without_overload_returns_property_struct() {
        let mut object = ObjectInstance::new("NoIdx".to_string());
        object.properties.insert("x".to_string(), Value::Num(4.0));
        let saved = block_on(saveobj_builtin(Value::Object(object))).expect("saveobj");
        let Value::Struct(st) = saved else {
            panic!("expected struct payload");
        };
        assert_eq!(st.fields.get("x"), Some(&Value::Num(4.0)));
    }

    #[test]
    fn saveobj_dispatches_registered_method() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let mut object = ObjectInstance::new("OverIdx".to_string());
        object.properties.insert("k".to_string(), Value::Num(9.0));
        object
            .properties
            .insert("nargs".to_string(), Value::Num(5.0));

        let saved = block_on(saveobj_builtin(Value::Object(object))).expect("saveobj");
        let Value::Struct(st) = saved else {
            panic!("expected saveobj struct");
        };
        assert_eq!(st.fields.get("k"), Some(&Value::Num(9.0)));
        assert_eq!(
            st.fields.get("saved_by"),
            Some(&Value::String("OverIdx.saveobj".to_string()))
        );
    }

    #[test]
    fn loadobj_restores_serialized_envelope_through_registered_method() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let mut object = ObjectInstance::new("OverIdx".to_string());
        object.properties.insert("k".to_string(), Value::Num(11.0));

        let envelope =
            block_on(prepare_value_for_mat_save(Value::Object(object))).expect("prepare");
        let restored = block_on(restore_value_from_mat_load(envelope)).expect("restore");
        let Value::Object(restored_object) = restored else {
            panic!("expected restored object");
        };
        assert_eq!(restored_object.class_name, "OverIdx");
        assert_eq!(restored_object.properties.get("k"), Some(&Value::Num(11.0)));
        assert_eq!(
            restored_object.properties.get("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string()))
        );
    }

    #[test]
    fn mat_roundtrip_calls_saveobj_and_loadobj() {
        block_on(crate::register_test_classes_builtin()).expect("test classes");
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("object_roundtrip.mat");

        let mut object = ObjectInstance::new("OverIdx".to_string());
        object.properties.insert("k".to_string(), Value::Num(13.0));
        object
            .properties
            .insert("nargs".to_string(), Value::Num(2.0));

        let bytes = block_on(
            crate::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[(
                "obj".to_string(),
                Value::Object(object),
            )]),
        )
        .expect("encode");
        std::fs::write(&path, bytes).expect("write mat file");

        let entries =
            block_on(crate::builtins::io::mat::load::read_mat_file(&path)).expect("load mat file");
        let loaded = entries
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("obj entry");
        let Value::Object(restored_object) = loaded else {
            panic!("expected restored object");
        };
        assert_eq!(restored_object.class_name, "OverIdx");
        assert_eq!(restored_object.properties.get("k"), Some(&Value::Num(13.0)));
        assert_eq!(
            restored_object.properties.get("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string()))
        );

        let decoded = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(
            &std::fs::read(&path).unwrap(),
        )
        .expect("decode workspace bytes");
        let decoded_value = decoded
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("decoded obj entry");
        let Value::Object(decoded_object) = decoded_value else {
            panic!("expected decoded object");
        };
        assert_eq!(decoded_object.class_name, "OverIdx");
        assert_eq!(
            decoded_object.properties.get("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string()))
        );
    }
}
