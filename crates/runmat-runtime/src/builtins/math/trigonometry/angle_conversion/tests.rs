use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_builtins::{DEG2RAD_ERROR_INVALID_INPUT, RAD2DEG_ERROR_INVALID_INPUT};
use runmat_value::Value;

#[test]
fn registered_conversions_reject_resident_complex_integers_before_provider_lookup() {
    let _guard = test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for (builtin, error_descriptor) in [
        ("deg2rad", &DEG2RAD_ERROR_INVALID_INPUT),
        ("rad2deg", &RAD2DEG_ERROR_INVALID_INPUT),
    ] {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        assert!(runmat_accelerate_api::provider().is_none());
        let error = block_on(crate::call_builtin_async(
            builtin,
            &[Value::GpuTensor(allocation.handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), error_descriptor.identifier);
        assert!(error
            .message()
            .contains("complex numbers with integer types"));
        assert_eq!(allocation.probe.retirements(), 0);
        assert!(allocation.handle.allocation_lease().validate_live().is_ok());
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
