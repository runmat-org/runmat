//! Runtime binding for canonical object predicate.
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isobject",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::isobject"
)]
fn isobject_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(isobject_value(&value)))
}

pub(crate) fn isobject_value(value: &Value) -> bool {
    matches!(
        value,
        Value::ObjectArray(_)
            | Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::Future(_)
            | Value::Task(_)
            | Value::Pool(_)
            | Value::Job(_)
    ) || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_value::{
        CellArray, CharArray, Closure, HandleRef, IntValue, IntegerStorage, Listener, MException,
        ObjectInstance, Tensor,
    };

    fn handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    #[test]
    fn runmat_object_variants_report_true() {
        let object = Value::Object(ObjectInstance::new("pkg.Point".to_string()));
        let handle = Value::HandleObject(HandleRef {
            class_name: "pkg.PointHandle".into(),
            target: handle_target(),
            valid: true,
        });
        let listener = Value::Listener(Listener {
            id: 1,
            target: handle_target(),
            target_class_name: "pkg.Target".into(),
            event_name: "Changed".into(),
            callback: handle_target(),
            enabled: true,
            valid: true,
        });
        let class_ref = Value::ClassRef("pkg.Point".into());
        let exception = Value::MException(MException::new(
            "RunMat:Test".to_string(),
            "failure".to_string(),
        ));

        for value in [object, handle, listener, class_ref, exception] {
            assert_eq!(
                isobject_builtin(value).expect("isobject"),
                Value::Bool(true)
            );
        }
    }

    #[test]
    fn non_object_values_report_false() {
        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.insert("x", Value::Num(1.0));
        let values = vec![
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor")),
            Value::Bool(true),
            Value::String("text".to_string().into()),
            Value::CharArray(CharArray::new_row("text")),
            Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell")),
            Value::Struct(st.finish()),
            Value::FunctionHandle("sin".into()),
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "anon".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: Vec::new().into(),
            })),
        ];

        for value in values {
            assert_eq!(
                isobject_builtin(value).expect("isobject"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn all_integer_classes_report_false() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            assert_eq!(
                isobject_builtin(Value::Int(value)).expect("isobject"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn only_explicit_resident_integer_reports_as_gpuarray_object() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let caller_alias = handle.clone();
            let original = handle.wire_descriptor();
            let lease = handle.allocation_lease().downgrade();
            provider.reset_telemetry();
            assert_eq!(
                isobject_builtin(Value::GpuTensor(handle.clone())).unwrap(),
                Value::Bool(false)
            );
            let explicit = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit immutable view");
            drop(handle);
            let handle = explicit;
            assert_eq!(
                isobject_builtin(Value::GpuTensor(handle)).unwrap(),
                Value::Bool(true)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(caller_alias.wire_descriptor(), original);
            assert!(lease.acquire().is_some());
            drop(caller_alias);
            assert!(lease.acquire().is_none());
        });
    }
}
