use super::super::*;

pub(super) async fn restore_envelope(
    payload: RootedValue,
    class_name: String,
    kind: String,
    had_saveobj: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<RootedValue> {
    if had_saveobj {
        if let Some(restored) =
            call_loadobj_if_available(&class_name, payload.value().clone(), runtime).await?
        {
            return Ok(restored);
        }
    }
    if matches!(payload.value(), Value::Struct(_)) {
        return payload.try_map_owned(
            |payload| {
                let Value::Struct(fields) = payload else {
                    unreachable!("checked above")
                };
                let mut object = ObjectInstance::new(class_name);
                object.properties = fields.fields.into_iter().collect();
                let object = Value::Object(object);
                match kind.as_str() {
                    SERIALIZED_KIND_VALUE => Ok(object),
                    SERIALIZED_KIND_HANDLE => {
                        let class_name =
                            object_class_name(&object).expect("constructed object class");
                        let target = runmat_gc::gc_allocate(object).map_err(|error| {
                            serialization_error(
                                LOADOBJ_METHOD.display_name(),
                                &SERIALIZATION_ERROR_ARGUMENT,
                                format!("failed to restore serialized handle object: {error}"),
                            )
                        })?;
                        Ok(Value::HandleObject(runmat_value::HandleRef {
                            class_name,
                            target,
                            valid: true,
                        }))
                    }
                    _ => Err(serialization_error(
                        LOADOBJ_METHOD.display_name(),
                        &SERIALIZATION_ERROR_ARGUMENT,
                        format!("unsupported serialized object kind '{kind}'"),
                    )),
                }
            },
            |error| error.into_runtime_error(),
        );
    }
    Ok(payload)
}
