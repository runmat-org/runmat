use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn strict_mode_rejects_resident_input_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 99,
        buffer_id: 77,
        descriptor: Default::default(),
    });
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
fn actual_wgpu_input_is_gathered_into_a_typed_host_table() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _accel_guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let input = Tensor::new_integer(IntegerStorage::I32(vec![-3, 7]), vec![1, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
    let output = table(call(Value::GpuTensor(handle), Vec::new()).unwrap());
    let variables = table_variables(&output).unwrap();
    assert!(matches!(
        &variables.fields["Var1"],
        Value::Tensor(column)
            if column.integer_storage() == Some(&IntegerStorage::I32(vec![-3, 7]))
    ));
}
