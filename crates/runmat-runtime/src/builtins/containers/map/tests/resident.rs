use super::*;

#[test]
fn nested_resident_inputs_are_gated_before_provider_access() {
    use runmat_value::{Closure, ObjectArray, StructArray};

    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let resident = Value::GpuTensor(allocation.handle.clone());
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("payload", resident.clone());
    let mut object = runmat_value::UnboundedObjectBuilder::new("TestResidentContainer");
    object.insert("payload".into(), resident.clone());
    let object = object.finish();
    let structure = structure.finish();
    let nested = [
        Value::Struct(structure.clone()),
        Value::StructArray(
            StructArray::new(vec![structure.clone(), structure], vec![1, 2]).unwrap(),
        ),
        Value::Object(object.clone()),
        Value::ObjectArray(
            ObjectArray::row("TestResidentContainer", vec![Value::Object(object)]).unwrap(),
        ),
        Value::Closure(Closure::from_data(runmat_value::ClosureData {
            lexical_environment: None,
            function_name: "captured".into(),
            binding: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
            captures: vec![resident.clone()].into(),
        })),
        Value::output_list(vec![resident.clone()]),
    ];
    for value in nested {
        assert!(contains_resident_value(&value));
        let error = ensure_resident_extension(
            &value,
            &MAP_RESIDENT_CONSTRUCTOR_EXTENSION,
            BUILTIN_CONSTRUCTOR,
        )
        .unwrap_err();
        assert!(error.identifier().is_some_and(|identifier| {
            identifier.starts_with("RunMat:compatibility:ContainersMapResident")
        }));
    }
    assert_eq!(allocation.probe.retirements(), 0);
    drop(resident);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn resident_extensions_gate_before_provider_access_for_integer_surfaces() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let char_map = containers_map_builtin(Vec::new()).unwrap();
    let checks = [
        block_on(super::super::containers_map_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                resident.clone(),
                Value::Num(1.0),
            ]),
        ))
        .unwrap_err(),
        block_on(super::super::containers_map_is_key(
            char_map.clone(),
            resident.clone(),
        ))
        .unwrap_err(),
        block_on(super::super::containers_map_remove(
            char_map.clone(),
            resident.clone(),
        ))
        .unwrap_err(),
        block_on(super::super::containers_map_values(
            char_map.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![crate::make_cell(
                vec![resident.clone()],
                1,
                1,
            )
            .unwrap()]),
        ))
        .unwrap_err(),
        containers_map_subsref(
            char_map.clone(),
            "()".to_string(),
            crate::make_cell(vec![resident.clone()], 1, 1).unwrap(),
        )
        .unwrap_err(),
        containers_map_subsasgn(
            char_map,
            "()".to_string(),
            crate::make_cell(vec![Value::from("x")], 1, 1).unwrap(),
            resident,
        )
        .unwrap_err(),
    ];
    for error in checks {
        assert!(error.identifier().is_some_and(
            |identifier| identifier.starts_with("RunMat:compatibility:ContainersMapResident")
        ));
    }
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn map_constructor_gathers_gpu_values() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let keys = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3, 1];
        let view = runmat_accelerate_api::HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let values = crate::make_cell(vec![Value::GpuTensor(handle)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![
            keys,
            values,
            Value::from("UniformValues"),
            Value::Bool(false),
        ])
        .expect("map");
        let payload = crate::make_cell(vec![Value::from("alpha")], 1, 1).unwrap();
        let value = containers_map_subsref(map, "()".to_string(), payload).expect("lookup");
        match value.value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), shape);
                assert_eq!(t.materialize_f64(), data);
            }
            other => panic!("expected tensor, got {:?}", other),
        }
    });
}

#[test]
fn map_resident_integer_keys_and_values_gather_exactly() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = 9_007_199_254_740_993_u64;
        let key_tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let key_handle = gpu_helpers::upload_tensor(provider, &key_tensor).expect("key upload");
        let map = containers_map_builtin(vec![Value::GpuTensor(key_handle), Value::Num(7.0)])
            .expect("resident key constructor");
        let payload = crate::make_cell(vec![Value::Int(IntValue::U64(wide))], 1, 1).unwrap();
        assert_eq!(
            containers_map_subsref(map, "()".to_string(), payload).unwrap(),
            Value::Num(7.0)
        );

        let value_tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![2, 1]).unwrap();
        let value_handle =
            gpu_helpers::upload_tensor(provider, &value_tensor).expect("value upload");
        let keys = crate::make_cell(vec![Value::from("wide")], 1, 1).unwrap();
        let values = crate::make_cell(vec![Value::GpuTensor(value_handle)], 1, 1).unwrap();
        let map = containers_map_builtin(vec![
            keys,
            values,
            Value::from("UniformValues"),
            Value::Bool(false),
        ])
        .expect("resident value constructor");
        let payload = crate::make_cell(vec![Value::from("wide")], 1, 1).unwrap();
        let gathered = containers_map_subsref(map, "()".to_string(), payload).unwrap();
        let Value::Tensor(gathered) = gathered.value() else {
            panic!("expected exact integer tensor value");
        };
        assert_eq!(gathered.integer_storage(), value_tensor.integer_storage());
    });
}
