use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};

#[test]
fn prepared_input_cancellation_retires_only_its_last_reference() {
    use std::future::Future;
    for retain_output in [false, true] {
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let output = retain_output.then(|| allocation.handle.clone());
        let prepared = PreparedInput {
            handle: allocation.handle,
        };
        let mut future = Box::pin(async move {
            std::future::pending::<()>().await;
            prepared.handle
        });
        assert!(future
            .as_mut()
            .poll(&mut std::task::Context::from_waker(
                &futures::task::noop_waker(),
            ))
            .is_pending());
        assert_eq!(allocation.probe.retirements(), 0);
        drop(future);
        assert_eq!(allocation.probe.retirements(), usize::from(!retain_output));
        drop(output);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn preparing_a_resident_input_retains_the_existing_lease() {
    let provider = crate::simple_provider::InProcessProvider::new();
    let handle = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2],
        })
        .unwrap();
    let weak = handle.downgrade();
    let input = Value::GpuTensor(handle);
    let prepared =
        ensure_gpu_tensor(&provider, &input, FusionPlacementObserver::new(None)).unwrap();
    let Value::GpuTensor(original) = &input else {
        unreachable!()
    };
    assert!(prepared.shares_allocation_with(original));
    drop(input);
    assert!(weak.acquire().is_ok());
    drop(prepared);
    assert!(weak.acquire().is_err());
}
