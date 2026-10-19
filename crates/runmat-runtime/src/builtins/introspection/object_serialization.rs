//! Object save/load customization hooks.

use crate::execution::{RootedValue, RootedValueLease, RootedValuesBuilder};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, StructValue, Value};

mod assembly;
mod mat_load;
mod prepare;
mod restore;
pub(crate) use mat_load::{classify_mat_load_restoration, MatLoadRestoration};

pub(crate) const SAVEOBJ_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("saveobj");
pub(crate) const LOADOBJ_METHOD: runmat_types::StaticMethodName =
    runmat_types::StaticMethodName::new("loadobj");

const SERIALIZED_CLASS_FIELD: &str = "__runmat_serialized_object_class__";
const SERIALIZED_PAYLOAD_FIELD: &str = "__runmat_serialized_object_payload__";
const SERIALIZED_KIND_FIELD: &str = "__runmat_serialized_object_kind__";
const SERIALIZED_HAD_SAVEOBJ_FIELD: &str = "__runmat_serialized_object_had_saveobj__";
const SERIALIZED_KIND_VALUE: &str = "value";
const SERIALIZED_KIND_HANDLE: &str = "handle";
const MAX_SERIALIZATION_DEPTH: usize = 32;

pub use runmat_builtins::{
    LOADOBJ_DESCRIPTOR, LOADOBJ_EXTENSIONS, LOADOBJ_INTEGER_CAPABILITIES,
    LOADOBJ_PASSTHROUGH_EXTENSION, SAVEOBJ_DESCRIPTOR, SAVEOBJ_INTEGER_AUDIT,
    SERIALIZATION_ERROR_ARGUMENT, SERIALIZATION_ERROR_DISPATCH, SERIALIZATION_ERROR_RECURSION,
};

fn serialization_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(builtin, descriptor, detail.into())
}

fn object_class_name(value: &Value) -> Option<runmat_types::ClassIdentity> {
    crate::object_receiver_class_name(value)
}

fn dispatch_registered_method<'a>(
    class_name: &'a runmat_types::ClassIdentity,
    method_name: runmat_types::StaticMethodName,
    args: Vec<Value>,
    builtin: &'static str,
    selected_runtime: Option<&crate::context::RuntimeContext>,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = crate::BuiltinResult<Option<RootedValue>>> + 'a>,
> {
    // The callback's inputs must remain rooted even if the admitted future is
    // retained unpolled, cancelled, or dropped by its caller.
    let runtime = selected_runtime
        .cloned()
        .or_else(crate::context::legacy::active);
    let args_root = match RootedValueLease::for_values(args.iter()) {
        Ok(root) => root,
        Err(error) => {
            return crate::execution::rooted_continuation::admit_rooted_future(
                runtime.as_ref(),
                builtin,
                crate::builtin::builtin_control_error,
                std::future::ready(Err(error.into_runtime_error())),
            );
        }
    };
    crate::execution::rooted_continuation::admit_rooted_future_with(
        runtime,
        builtin,
        crate::builtin::builtin_control_error,
        move |runtime| async move {
            let _args_root = args_root;
            if let Some((method, owner)) =
                crate::class_registry::lookup_method(class_name, &method_name.owned())
            {
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
                    let (identity, fallback_policy) =
                        crate::callable_identity_for_handle_name(&candidate);
                    match crate::dispatch_callable_with_policy_in_context(
                        runtime.as_ref(),
                        identity,
                        fallback_policy,
                        args.clone(),
                        1,
                    )
                    .await
                    {
                        Ok(value) => return Ok(Some(value)),
                        Err(err) if crate::is_undefined_function_error(&err) => {
                            undefined = Some(err)
                        }
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
                match crate::dispatch_object_external_member_in_context(
                    runtime.as_ref(),
                    class_name.clone(),
                    method_name.display_name(),
                    args,
                    1,
                )
                .await
                {
                    Ok(value) => Ok(Some(value)),
                    Err(err) if crate::is_undefined_function_error(&err) => Ok(None),
                    Err(err) => Err(err),
                }
            } else {
                Ok(None)
            }
        },
    )
}

async fn call_saveobj_if_available(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Option<RootedValue>> {
    let Some(class_name) = object_class_name(&value) else {
        return Ok(None);
    };
    dispatch_registered_method(
        &class_name,
        SAVEOBJ_METHOD,
        vec![value],
        SAVEOBJ_METHOD.display_name(),
        runtime,
    )
    .await
}

async fn call_loadobj_if_available(
    class_name: &str,
    payload: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<Option<RootedValue>> {
    let identity = runmat_types::ClassIdentity::new(class_name).map_err(|error| {
        serialization_error(
            LOADOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("invalid serialized class identity: {error}"),
        )
    })?;
    dispatch_registered_method(
        &identity,
        LOADOBJ_METHOD,
        vec![payload],
        LOADOBJ_METHOD.display_name(),
        runtime,
    )
    .await
}

fn object_properties_payload(value: &Value) -> crate::BuiltinResult<StructValue> {
    match value {
        Value::Object(object) => {
            if object.has_native_payload() {
                return Err(serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "GC-owned native object payloads require a typed serializer",
                ));
            }
            let mut payload = StructValue::new();
            for (name, property) in &object.properties {
                payload.insert(name.clone(), property.clone());
            }
            Ok(payload)
        }
        Value::HandleObject(handle) => runmat_gc::gc_with_value(&handle.target, |target| {
            if let Value::Object(object) = target {
                if object.has_native_payload() {
                    return Err(serialization_error(
                        SAVEOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        "GC-owned native object payloads require a typed serializer",
                    ));
                }
                let mut payload = StructValue::new();
                for (name, property) in &object.properties {
                    payload.insert(name.clone(), property.clone());
                }
                Ok(payload)
            } else {
                Err(serialization_error(
                    SAVEOBJ_METHOD.display_name(),
                    &SERIALIZATION_ERROR_ARGUMENT,
                    "handle target is not an object",
                ))
            }
        })
        .map_err(|err| {
            serialization_error(
                SAVEOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_ARGUMENT,
                format!("failed to read handle target: {err}"),
            )
        })?,
        other => Err(serialization_error(
            SAVEOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

fn ensure_no_native_payload(value: &Value) -> crate::BuiltinResult<()> {
    let native = match value {
        Value::Object(object) => object.has_native_payload(),
        Value::HandleObject(handle) => runmat_gc::gc_with_value(
            &handle.target,
            |target| matches!(target, Value::Object(object) if object.has_native_payload()),
        )
        .map_err(|error| {
            serialization_error(
                SAVEOBJ_METHOD.display_name(),
                &SERIALIZATION_ERROR_ARGUMENT,
                format!("failed to read handle target: {error}"),
            )
        })?,
        _ => false,
    };
    if native {
        Err(serialization_error(
            SAVEOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            "GC-owned native object payloads require a typed serializer",
        ))
    } else {
        Ok(())
    }
}

fn serialized_object_envelope(
    class_name: runmat_types::ClassIdentity,
    kind: &'static str,
    had_saveobj: bool,
    payload: Value,
) -> Value {
    let mut st = StructValue::new();
    st.insert(
        SERIALIZED_CLASS_FIELD,
        Value::String(class_name.display_name().to_owned()),
    );
    st.insert(SERIALIZED_KIND_FIELD, Value::String(kind.to_string()));
    st.insert(SERIALIZED_HAD_SAVEOBJ_FIELD, Value::Bool(had_saveobj));
    st.insert(SERIALIZED_PAYLOAD_FIELD, payload);
    Value::Struct(st)
}

fn into_serialized_envelope(value: Value) -> Result<(String, String, bool, Value), Value> {
    let Value::Struct(mut st) = value else {
        return Err(value);
    };
    let Some(class_name) = st
        .fields
        .get(SERIALIZED_CLASS_FIELD)
        .and_then(|value| String::try_from(value).ok())
    else {
        return Err(Value::Struct(st));
    };
    if !st.fields.contains_key(SERIALIZED_PAYLOAD_FIELD) {
        return Err(Value::Struct(st));
    }
    let Some(kind) = st
        .fields
        .get(SERIALIZED_KIND_FIELD)
        .and_then(|value| String::try_from(value).ok())
    else {
        return Err(Value::Struct(st));
    };
    let Some(had_saveobj) = st
        .fields
        .get(SERIALIZED_HAD_SAVEOBJ_FIELD)
        .and_then(|value| {
            if let Value::Bool(value) = value {
                Some(*value)
            } else {
                None
            }
        })
    else {
        return Err(Value::Struct(st));
    };
    let payload = st
        .fields
        .shift_remove(SERIALIZED_PAYLOAD_FIELD)
        .expect("validated serialized payload field");
    Ok((class_name, kind, had_saveobj, payload))
}

fn root_value(value: Value) -> crate::BuiltinResult<RootedValue> {
    RootedValue::new(value).map_err(|error| error.into_runtime_error())
}

pub(crate) async fn prepare_value_for_mat_save(value: Value) -> crate::BuiltinResult<RootedValue> {
    prepare_value_for_mat_save_in_context(value, None).await
}

pub(crate) async fn prepare_value_for_mat_save_in_context(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    prepare::prepare_value_for_save_in_context(value, runtime).await
}

pub(crate) async fn restore_value_from_mat_load(value: Value) -> crate::BuiltinResult<RootedValue> {
    restore_value_from_mat_load_in_context(value, None).await
}

pub(crate) async fn restore_value_from_mat_load_in_context(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    restore::restore_value_after_load_in_context(value, runtime).await
}

#[runtime_builtin(
    name = "saveobj",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn saveobj_builtin(value: Value) -> crate::BuiltinResult<RootedValue> {
    let _input_root =
        RootedValueLease::for_traced(&value).map_err(|error| error.into_runtime_error())?;
    match value {
        receiver @ (Value::Object(_) | Value::HandleObject(_)) => {
            if let Some(saved) = call_saveobj_if_available(receiver.clone(), None).await? {
                Ok(saved)
            } else {
                root_value(Value::Struct(object_properties_payload(&receiver)?))
            }
        }
        other => Err(serialization_error(
            SAVEOBJ_METHOD.display_name(),
            &SERIALIZATION_ERROR_ARGUMENT,
            format!("expected object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "loadobj",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::object_serialization"
)]
pub async fn loadobj_builtin(value: Value) -> crate::BuiltinResult<RootedValue> {
    if !matches!(
        value,
        Value::Object(_) | Value::HandleObject(_) | Value::Struct(_) | Value::StructArray(_)
    ) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOADOBJ_PASSTHROUGH_EXTENSION,
            LOADOBJ_METHOD.display_name(),
        )?;
    }
    restore_value_from_mat_load(value).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, StructArray, Tensor};

    use tempfile::tempdir;

    #[test]
    fn mat_save_rejects_native_object_payload_instead_of_erasing_it() {
        let object = ObjectInstance::new("containers.Map").with_containers_map_payload(
            runmat_value::ContainersMapStorage::new(
                runmat_value::ContainersMapKeyType::Char,
                runmat_value::ContainersMapValueType::Any,
                false,
            ),
        );
        let error = block_on(prepare_value_for_mat_save(Value::Object(object)))
            .expect_err("native payload needs an explicit serializer");
        assert!(error.message.contains("native object payloads"));
    }

    fn structure(field: &str, value: Value) -> StructValue {
        let mut structure = StructValue::new();
        structure.insert(field, value);
        structure
    }

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
        let Value::Struct(restored) = restored.value() else {
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
    fn mat_object_recursion_preserves_empty_and_nd_structure_arrays() {
        let empty = StructArray::with_fields(vec!["payload".into()], Vec::new(), vec![0, 2, 3])
            .expect("empty structure array");
        let saved = block_on(prepare_value_for_mat_save(Value::StructArray(empty)))
            .expect("prepare empty array");
        let Value::StructArray(saved) = saved.value() else {
            panic!("expected typed empty structure array");
        };
        assert_eq!(saved.shape(), [0, 2, 3]);
        assert_eq!(
            saved.field_names().cloned().collect::<Vec<_>>(),
            ["payload"]
        );

        let elements = (0..4)
            .map(|index| structure("payload", Value::Int(IntValue::U64(index))))
            .collect();
        let array = StructArray::with_fields(vec!["payload".into()], elements, vec![2, 1, 2])
            .expect("N-D structure array");
        let prepared =
            block_on(prepare_value_for_mat_save(Value::StructArray(array))).expect("prepare");
        let restored =
            block_on(restore_value_from_mat_load(prepared.value().clone())).expect("restore");
        let Value::StructArray(restored) = restored.value() else {
            panic!("expected typed N-D structure array");
        };
        assert_eq!(restored.shape(), [2, 1, 2]);
        for (index, element) in restored.elements().enumerate() {
            assert_eq!(
                element.fields.get("payload"),
                Some(&Value::Int(IntValue::U64(index as u64)))
            );
        }
    }

    #[test]
    fn mat_object_recursion_restores_nested_value_and_handle_objects() {
        let mut value_object = ObjectInstance::new("NestedValue".to_string());
        value_object
            .properties
            .insert("payload".into(), Value::Int(IntValue::U64(u64::MAX)));

        let mut handle_object = ObjectInstance::new("NestedHandle".to_string());
        handle_object
            .properties
            .insert("payload".into(), Value::String("entry".into()));
        let target = runmat_gc::gc_allocate(Value::Object(handle_object)).expect("handle target");
        let handle = Value::HandleObject(runmat_value::HandleRef {
            class_name: "NestedHandle".into(),
            target,
            valid: true,
        });

        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", Value::Object(value_object)),
                structure("payload", handle),
            ],
            vec![2, 1],
        )
        .expect("structure array");
        let prepared =
            block_on(prepare_value_for_mat_save(Value::StructArray(array))).expect("prepare");
        let restored =
            block_on(restore_value_from_mat_load(prepared.value().clone())).expect("restore");
        let Value::StructArray(restored) = restored.value() else {
            panic!("expected typed structure array");
        };
        assert!(matches!(
            restored.get_linear(0).unwrap().fields.get("payload"),
            Some(Value::Object(object)) if object.class_name.display_name() == "NestedValue"
        ));
        let Some(Value::HandleObject(handle)) =
            restored.get_linear(1).unwrap().fields.get("payload")
        else {
            panic!("expected nested handle object");
        };
        assert_eq!(handle.class_name.display_name(), "NestedHandle");
        assert!(runmat_gc::gc_with_value(&handle.target, |target| {
            matches!(target, Value::Object(object)
                if object.properties.get("payload") == Some(&Value::String("entry".into())))
        })
        .expect("live handle target"));
    }

    #[test]
    fn mat_object_recursion_reports_nested_invalid_envelope_and_depth() {
        let invalid = serialized_object_envelope(
            "InvalidEnvelope".into(),
            "unsupported",
            false,
            Value::Struct(StructValue::new()),
        );
        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", invalid),
                structure("payload", Value::Num(1.0)),
            ],
            vec![2, 1],
        )
        .expect("structure array");
        let error = block_on(restore_value_from_mat_load(Value::StructArray(array)))
            .expect_err("invalid nested envelope must fail");
        assert_eq!(error.identifier(), SERIALIZATION_ERROR_ARGUMENT.identifier);

        let mut nested = Value::Num(1.0);
        for _ in 0..=MAX_SERIALIZATION_DEPTH {
            nested = Value::Struct(structure("next", nested));
        }
        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", nested),
                structure("payload", Value::Num(2.0)),
            ],
            vec![2, 1],
        )
        .expect("structure array");
        let error = block_on(prepare_value_for_mat_save(Value::StructArray(array)))
            .expect_err("nested depth must fail");
        assert_eq!(error.identifier(), SERIALIZATION_ERROR_RECURSION.identifier);
    }

    #[test]
    fn loadobj_plain_integer_passthrough_is_declared_and_mode_gated() {
        assert_eq!(LOADOBJ_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(LOADOBJ_EXTENSIONS.len(), 1);
        let value = Value::Int(IntValue::U64(u64::MAX));
        let runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(
            block_on(loadobj_builtin(value.clone()))
                .expect("RunMat passthrough")
                .value(),
            &value
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
    fn loadobj_accepts_typed_struct_arrays_in_compatibility_mode() {
        let array = StructArray::with_fields(
            vec!["payload".into()],
            vec![
                structure("payload", Value::Num(1.0)),
                structure("payload", Value::Num(2.0)),
            ],
            vec![2, 1],
        )
        .unwrap();
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let restored = block_on(loadobj_builtin(Value::StructArray(array.clone())))
            .expect("structure arrays are part of the compatible loadobj domain");
        drop(strict);
        assert_eq!(restored.value(), &Value::StructArray(array));
    }

    #[test]
    fn saveobj_without_overload_returns_property_struct() {
        let mut object = ObjectInstance::new("NoIdx".to_string());
        object.properties.insert("x".to_string(), Value::Num(4.0));
        let saved = block_on(saveobj_builtin(Value::Object(object))).expect("saveobj");
        let Value::Struct(st) = saved.value() else {
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
        let Value::Struct(st) = saved.value() else {
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
        let restored =
            block_on(restore_value_from_mat_load(envelope.value().clone())).expect("restore");
        let Value::Object(restored_object) = restored.value() else {
            panic!("expected restored object");
        };
        assert_eq!(restored_object.class_name.display_name(), "OverIdx");
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
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("obj entry");
        let Value::Object(restored_object) = loaded else {
            panic!("expected restored object");
        };
        assert_eq!(restored_object.class_name.display_name(), "OverIdx");
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
            .value()
            .0
            .iter()
            .find(|(name, _)| name == "obj")
            .map(|(_, value)| value)
            .expect("decoded obj entry");
        let Value::Object(decoded_object) = decoded_value else {
            panic!("expected decoded object");
        };
        assert_eq!(decoded_object.class_name.display_name(), "OverIdx");
        assert_eq!(
            decoded_object.properties.get("loaded_by"),
            Some(&Value::String("OverIdx.loadobj".to_string()))
        );
    }
}
