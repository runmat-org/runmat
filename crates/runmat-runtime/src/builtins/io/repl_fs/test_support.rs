/// A genuine resident Double input for filesystem metadata-admission tests.
/// There is deliberately no registered provider: these tests must reject the
/// argument or a later invalid option before attempting any device read.
#[cfg(test)]
pub(crate) fn unregistered_numeric_allocation(
    shape: Vec<usize>,
) -> runmat_accelerate_api::test_support::TestAllocation {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    TestAllocationOwner::new()
        .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
        .expect("valid filesystem admission fixture")
}

#[cfg(test)]
mod tests {
    #[test]
    fn unregistered_admission_fixture_has_real_shared_allocation_lifetime() {
        let allocation = super::unregistered_numeric_allocation(vec![1, 2]);
        assert_eq!(allocation.handle.shape(), &[1, 2]);
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let borrowed = allocation.handle.clone();
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 0);
        borrowed.allocation_lease().validate_live().unwrap();
        drop(borrowed);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
