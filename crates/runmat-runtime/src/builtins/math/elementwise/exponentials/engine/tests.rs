use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage};

#[test]
fn complex_integer_host_rejection_uses_selected_operation_error() {
    for operation in [ExponentialOperation::Exp, ExponentialOperation::Expm1] {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![9_223_372_036_854_775_808]),
            IntegerStorage::U64(vec![1]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).unwrap();
        for error in [
            block_on(execute_non_table(
                None,
                operation,
                Value::ComplexTensor(tensor.clone()),
            ))
            .unwrap_err(),
            super::super::host::evaluate(operation, Value::ComplexTensor(tensor)).unwrap_err(),
        ] {
            assert_eq!(error.identifier(), operation.invalid_input().identifier);
            assert!(error
                .message()
                .contains("complex numbers with integer types"));
        }
    }
}

#[test]
fn resident_complex_integer_rejection_precedes_provider_lookup() {
    let _guard = test_support::accel_test_lock();
    for operation in [ExponentialOperation::Exp, ExponentialOperation::Expm1] {
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::U64,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        assert!(runmat_accelerate_api::provider().is_none());
        let error = block_on(execute_non_table(
            None,
            operation,
            Value::GpuTensor(allocation.handle.clone()),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), operation.invalid_input().identifier);
        assert_eq!(allocation.probe.retirements(), 0);
        assert!(allocation.handle.allocation_lease().validate_live().is_ok());
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
