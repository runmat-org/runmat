use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn strict_mode_rejects_nested_resident_before_provider_access() {
    use runmat_value::{CellArray, Closure, ObjectInstance};
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let resident = Value::GpuTensor(
        allocation_owner
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("data", resident.clone());
    let mut object = ObjectInstance::new("GatherFixture");
    object.replace_property_unbounded("data".into(), resident.clone());
    for input in [
        Value::Cell(
            CellArray::new_with_shape(vec![Value::Struct(structure.finish())], vec![1, 1]).unwrap(),
        ),
        Value::Object(object),
        Value::Closure(Closure::new(
            "fixture".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![resident],
            None,
        )),
    ] {
        let error = call(input, Vec::new()).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:CombinationsResidentInputExtension")
        );
    }
}

#[test]
fn nested_resident_gather_preserves_exact_class_shape_and_borrowed_buffer() {
    use runmat_value::CellArray;
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
            vec![2, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("data", Value::GpuTensor(handle.clone()));
        let nested = Value::Cell(
            CellArray::new_with_shape(vec![Value::Struct(structure.finish())], vec![1, 1]).unwrap(),
        );
        let output = table(call(nested, Vec::new()).unwrap());
        let variables = table_variables(&output).unwrap();
        let Value::Cell(column) = &variables.fields["Var1"] else {
            panic!("cell column")
        };
        assert_eq!(column.shape, vec![1, 1]);
        let Value::Struct(structure) = &column.data()[0] else {
            panic!("nested structure")
        };
        assert_eq!(structure.fields["data"], Value::Tensor(input.clone()));
        assert_eq!(
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
                handle.clone()
            )))
            .unwrap(),
            Value::Tensor(input)
        );
        handle.release().unwrap();
    });
}

#[test]
fn strict_mode_rejects_resident_input_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let error = call(resident, Vec::new()).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:CombinationsResidentInputExtension")
    );
}

#[test]
fn resident_integer_input_is_gathered_exactly_into_the_host_table() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let output = table(call(Value::GpuTensor(handle), Vec::new()).unwrap());
        let variables = table_variables(&output).unwrap();
        let Value::Tensor(column) = &variables.fields["Var1"] else {
            panic!("expected integer column")
        };
        assert_eq!(
            column.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a real WGPU device; run explicitly with --ignored"]
fn actual_wgpu_input_is_gathered_into_a_typed_host_table() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _accel_guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU device required for this explicit hardware witness");
    let provider: &dyn runmat_accelerate_api::AccelProvider = provider;
    let input = Tensor::new_integer(IntegerStorage::I32(vec![-3, 7]), vec![1, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
    let output = table(call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap());
    let variables = table_variables(&output).unwrap();
    assert!(matches!(
        &variables.fields["Var1"],
        Value::Tensor(column)
            if column.integer_storage() == Some(&IntegerStorage::I32(vec![-3, 7]))
    ));
    handle.release().unwrap();
}
