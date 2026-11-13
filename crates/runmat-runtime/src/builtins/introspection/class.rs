//! Runtime binding for canonical class metadata query.
#[cfg(test)]
use crate::builtins::introspection::class_identity::class_identity_for_value;
use crate::builtins::introspection::class_identity::class_name_for_value;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "class",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::class"
)]
fn class_builtin(value: Value) -> crate::BuiltinResult<String> {
    Ok(class_name_for_value(&value))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, IntegerComplexStorage,
        IntegerStorage, Listener, LogicalArray, MException, ObjectInstance, StringArray,
        SymbolicExpr, Tensor,
    };

    fn test_handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_double_for_numeric_scalars() {
        let name = class_builtin(Value::Num(std::f64::consts::PI)).expect("class");
        assert_eq!(name, "double");
    }

    #[test]
    fn nested_class_query_uses_string_value_carrier() {
        let inner = class_builtin(Value::Int(IntValue::U64(u64::MAX))).expect("inner class");
        let value = Value::from(inner);
        assert!(matches!(value, Value::String(_)));
        assert_eq!(class_builtin(value).expect("outer class"), "string");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_integer_type_names() {
        let name = class_builtin(Value::Int(IntValue::I32(12))).expect("class");
        assert_eq!(name, "int32");
    }

    #[test]
    fn class_identity_remains_typed_until_the_display_boundary() {
        let value = Value::Int(IntValue::I32(12));
        assert_eq!(
            class_identity_for_value(&value),
            runmat_types::standard::INT32.owned()
        );
        assert_eq!(class_name_for_value(&value), "int32");
    }

    #[test]
    fn class_reports_exact_integer_tensor_storage_type() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .expect("uint64 tensor");

        assert_eq!(class_name_for_value(&Value::Tensor(tensor)), "uint64");
    }

    #[test]
    fn class_reports_exact_integer_sparse_storage_type() {
        let sparse = runmat_value::SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");

        assert_eq!(class_name_for_value(&Value::SparseTensor(sparse)), "uint64");
    }

    #[test]
    fn class_reports_exact_integer_type_for_every_typed_complex_class() {
        let cases = [
            (
                "int8",
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I8(vec![2]),
            ),
            (
                "int16",
                IntegerStorage::I16(vec![-3]),
                IntegerStorage::I16(vec![4]),
            ),
            (
                "int32",
                IntegerStorage::I32(vec![-5]),
                IntegerStorage::I32(vec![6]),
            ),
            (
                "int64",
                IntegerStorage::I64(vec![-7]),
                IntegerStorage::I64(vec![8]),
            ),
            (
                "uint8",
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![2]),
            ),
            (
                "uint16",
                IntegerStorage::U16(vec![3]),
                IntegerStorage::U16(vec![4]),
            ),
            (
                "uint32",
                IntegerStorage::U32(vec![5]),
                IntegerStorage::U32(vec![6]),
            ),
            (
                "uint64",
                IntegerStorage::U64(vec![7]),
                IntegerStorage::U64(vec![8]),
            ),
        ];

        for (expected, real, imag) in cases {
            let storage = IntegerComplexStorage::new(real, imag).expect("matching components");
            let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");
            assert_eq!(
                class_name_for_value(&Value::ComplexTensor(tensor)),
                expected
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_sym_for_symbolic_values() {
        let name = class_builtin(Value::Symbolic(SymbolicExpr::variable("x"))).expect("class");

        assert_eq!(name, "sym");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_expected_names_for_core_types() {
        let logical_scalar = Value::Bool(true);
        assert_eq!(class_name_for_value(&logical_scalar), "logical");

        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8, 1u8, 1u8], vec![2, 2]).expect("logical array"),
        );
        assert_eq!(class_name_for_value(&logical_array), "logical");

        let string_scalar = Value::String("hello".to_string().into());
        assert_eq!(class_name_for_value(&string_scalar), "string");

        let string_array = Value::StringArray(
            StringArray::new(vec!["entry-a".into(), "entry-b".into()], vec![1, 2])
                .expect("string array"),
        );
        assert_eq!(class_name_for_value(&string_array), "string");

        let char_array = Value::CharArray(CharArray::new_row("abc"));
        assert_eq!(class_name_for_value(&char_array), "char");

        let real_tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        assert_eq!(class_name_for_value(&real_tensor), "double");

        let complex_scalar = Value::Complex(1.0, -1.0);
        assert_eq!(class_name_for_value(&complex_scalar), "double");

        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -3.0)], vec![2, 1]).expect("complex tensor"),
        );
        assert_eq!(class_name_for_value(&complex_tensor), "double");

        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).unwrap());
        assert_eq!(class_name_for_value(&cell), "cell");

        let mut st = runmat_value::UnboundedStructValueBuilder::new();
        st.fields.insert("field".into(), Value::Num(42.0));
        let struct_value = Value::Struct(st.finish());
        assert_eq!(class_name_for_value(&struct_value), "struct");

        let func_handle = Value::FunctionHandle("sin".into());
        assert_eq!(class_name_for_value(&func_handle), "function_handle");

        let closure = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "anon".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![].into(),
        }));
        assert_eq!(class_name_for_value(&closure), "function_handle");

        let class_ref = Value::ClassRef("pkg.Point".into());
        assert_eq!(class_name_for_value(&class_ref), "meta.class");

        let exception = Value::MException(MException::new("id:err".into(), "fail".into()));
        assert_eq!(class_name_for_value(&exception), "MException");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_distinguishes_automatic_and_explicit_residency_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            assert_eq!(
                class_builtin(Value::GpuTensor(handle.clone())).expect("automatic class"),
                "double"
            );
            let explicit = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit view");
            let lease = handle.allocation_lease().downgrade();
            assert_eq!(
                class_builtin(Value::GpuTensor(explicit)).expect("explicit class"),
                "gpuArray"
            );
            assert!(lease.acquire().is_some());
            drop(handle);
            assert!(lease.acquire().is_none());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_handle_class_identitys() {
        let fallback = HandleRef {
            class_name: runmat_types::standard::HANDLE.into(),
            target: test_handle_target(),
            valid: false,
        };
        let fallback_name = class_builtin(Value::HandleObject(fallback)).expect("class");
        assert_eq!(fallback_name, "handle");

        let handle = HandleRef {
            class_name: "MockHandle".into(),
            target: test_handle_target(),
            valid: true,
        };
        let name = class_builtin(Value::HandleObject(handle)).expect("class");
        assert_eq!(name, "MockHandle");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_object_and_listener_classes() {
        let object = ObjectInstance::new("pkg.Point");
        let obj_name = class_builtin(Value::Object(object)).expect("class object");
        assert_eq!(obj_name, "pkg.Point");

        let listener = Listener {
            id: 1,
            target: test_handle_target(),
            target_class_name: "EventTarget".into(),
            event_name: "changed".into(),
            callback: test_handle_target(),
            enabled: true,
            valid: true,
        };
        let listener_name = class_builtin(Value::Listener(listener)).expect("class listener");
        assert_eq!(listener_name, "event.listener");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn class_reports_gpuarray_with_wgpu_provider() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        // Attempt to register a WGPU provider; skip if the environment lacks a compatible adapter.
        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit GPU view");
        let name = class_builtin(Value::GpuTensor(handle)).expect("class");
        assert_eq!(name, "gpuArray");
    }
}
