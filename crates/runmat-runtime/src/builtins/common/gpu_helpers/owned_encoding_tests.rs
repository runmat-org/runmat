use super::*;
use runmat_accelerate_api::test_support::TestAllocationOwner;

#[test]
fn logical_u8_owner_preserves_physical_type_and_logical_host_class() {
    let owner = TestAllocationOwner::new();
    let allocation = owner.allocate_logical(vec![0, 1], vec![1, 2]).unwrap();
    assert_eq!(
        expected_handle_numeric_element_type(&allocation.handle).unwrap(),
        NumericElementType::U8
    );
    let value = value_from_numeric_download(
        HostNumericTensorOwned {
            data: HostNumericDataOwned::U8(vec![0, 1]),
            shape: vec![1, 2],
            storage: GpuTensorStorage::Real,
        },
        runmat_accelerate_api::handle_is_logical(&allocation.handle),
    )
    .unwrap();
    let Value::LogicalArray(value) = value else {
        panic!("logical producer must remain logical")
    };
    assert_eq!(value.data, [0, 1]);
    assert_eq!(value.shape, [1, 2]);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn numeric_u8_download_does_not_acquire_logical_interpretation() {
    let value = value_from_numeric_download(
        HostNumericTensorOwned {
            data: HostNumericDataOwned::U8(vec![0, 2]),
            shape: vec![1, 2],
            storage: GpuTensorStorage::Real,
        },
        false,
    )
    .unwrap();
    let Value::Tensor(value) = value else {
        panic!("numeric producer must remain numeric")
    };
    assert!(
        matches!(value.integer_storage(), Some(IntegerStorage::U8(values)) if values == &[0, 2])
    );
}
