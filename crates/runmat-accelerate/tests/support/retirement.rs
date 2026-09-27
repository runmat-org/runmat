//! Consume a complete set of test-owned views and verify physical retirement.
//! Weak leases observe the actual allocation owner, not diagnostic IDs.
use runmat_accelerate_api::GpuTensorHandle;

pub fn release_all(handles: impl IntoIterator<Item = GpuTensorHandle>) {
    let handles: Vec<_> = handles.into_iter().collect();
    let allocations: Vec<_> = handles
        .iter()
        .map(|handle| {
            (
                handle.allocation_identity(),
                handle.allocation_lease().downgrade(),
            )
        })
        .collect();
    for handle in handles {
        handle
            .release()
            .expect("test allocation release must succeed");
    }
    for (identity, weak) in allocations {
        assert!(
            weak.acquire().is_none(),
            "allocation {identity:?} still has an owner"
        );
    }
}
