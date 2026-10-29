//! Required-device tests: adapter absence is never reported as a pass.
#![cfg(not(target_arch = "wasm32"))]
use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
use super::*;
use runmat_accelerate_api::{AccelProvider, HostTensorView};

fn provider() -> WgpuProvider {
    WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU adapter")
}
fn upload(provider: &WgpuProvider) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap()
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn issued_probe_tensor_ticket_retires_only_with_canonical_last_lease() {
    use runmat_accelerate_api::observation::{ProviderObservationStatus, ProviderOperationReceipt};
    let provider = provider();
    let borrowed = upload(&provider);
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(
            runmat_accelerate_api::observation::ProviderResourcePhasePlan::new([1024; 4]).unwrap(),
        )
        .unwrap();
    for _ in 0..4 {
        scope.begin_next_phase().unwrap();
    }
    let probe_owned = upload(&provider);
    let alias = provider.reshape_exec(&probe_owned, &[2, 1]).unwrap();
    let ticket = scope
        .issue_resource_ticket(probe_owned.physical().requested_bytes())
        .unwrap();
    provider
        .allocations
        .attach_probe_resource_ticket(&probe_owned, ticket)
        .unwrap();
    scope.finish().unwrap();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    drop(probe_owned);
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    drop(alias);
    assert!(matches!(
        receipt.status(),
        ProviderObservationStatus::Completed { .. }
    ));
    let resources = receipt.take_completed().unwrap().resources.unwrap();
    assert_eq!(resources.issued, 1);
    assert_eq!(resources.retired, 1);
    assert_eq!(resources.requested_live_bytes, 0);
    assert_eq!(
        resources.requested_peak_bytes,
        borrowed.physical().requested_bytes()
    );
    borrowed.allocation_lease().validate_live().unwrap();
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn views_and_exports_share_one_lease_and_facade_drop_does_not_invalidate_them() {
    let provider = provider();
    let state = super::super::owner_shared::OwnerShared::downgrade(&provider.allocations);
    let input = upload(&provider);
    let clone = input.clone();
    let alias = provider.reshape_exec(&input, &[2, 1]).unwrap();
    assert_eq!(input.view_identity(), clone.view_identity());
    assert_ne!(input.view_identity(), alias.view_identity());
    assert!(input.shares_allocation_with(&alias));
    assert_eq!(provider.allocations.record_count().unwrap(), 1);
    let export = provider.export_wgpu_buffer(&input).unwrap().unwrap();
    let wire = alias.wire_descriptor();
    let imported = provider.acquire_live(&wire).unwrap();
    assert!(alias.shares_allocation_with(&imported));
    drop(input);
    drop(clone);
    assert_eq!(
        futures::executor::block_on(provider.download(&alias))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
    drop(alias);
    drop(imported);
    assert_eq!(provider.allocations.record_count().unwrap(), 1);
    drop(provider);
    assert!(
        state.upgrade().is_some(),
        "export retains captured physical owner"
    );
    export
        .owning_handle()
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(export);
    assert!(
        state.upgrade().is_none(),
        "no facade/provenance/lease cycle"
    );
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn stale_foreign_and_modified_wire_descriptors_never_recreate_ownership() {
    let provider = provider();
    let other = self::provider();
    let input = upload(&provider);
    let mut wire = input.wire_descriptor();
    assert!(other.acquire_live(&wire).is_err());
    wire.view = wire.view.clone().with_logical(true);
    assert!(provider.acquire_live(&wire).is_err());
    wire = input.wire_descriptor();
    wire.allocation = runmat_accelerate_api::allocation::AllocationIdentity::new(
        input.device_id(),
        input.buffer_id(),
        2,
    );
    assert!(provider.acquire_live(&wire).is_err());
    wire = input.wire_descriptor();
    drop(input);
    assert!(provider.acquire_live(&wire).is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), 0);
}

#[cfg(panic = "unwind")]
#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn failed_retirement_retains_physical_record_without_pooling_or_reacquisition() {
    let provider = provider();
    let input = upload(&provider);
    let wire = input.wire_descriptor();
    let original = provider.get_entry_raw(&input).unwrap();
    let before = provider.allocations.pooled_bytes(provider.element_size);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| provider
            .bind_group_cache
            .poison_for_retirement_test()))
        .is_err()
    );
    let error = input.release().unwrap_err();
    assert!(error.physical.is_err());
    assert!(provider.acquire_live(&wire).is_err());
    {
        let records = provider.allocations.buffers.lock().unwrap();
        let retained = records.get(&wire.allocation.allocation_id()).unwrap();
        assert_eq!(
            retained.phase,
            Phase::Failed {
                stage: Stage::Cache,
                failure: Failure::Cache,
                disposition: Disposition::Retained
            }
        );
        assert!(runmat_accelerate_api::WgpuBufferOwner::ptr_eq(
            &original.buffer,
            &retained.entry.as_ref().unwrap().buffer
        ));
    }
    assert_eq!(
        provider.allocations.pooled_bytes(provider.element_size),
        before
    );
    drop(error);
    drop(provider);
    assert_eq!(
        runmat_accelerate_api::WgpuBufferOwner::strong_count(&original.buffer),
        1,
        "terminal teardown disposed the retained payload reference"
    );
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn actual_ndgrid_partial_registration_failure_rolls_back_only_fresh_outputs() {
    use runmat_accelerate_api::{ProviderNdgridAxis, ProviderNdgridRequest};
    let provider = provider();
    let x = upload(&provider);
    let y = upload(&provider);
    provider
        .allocations
        .next_id
        .store(u64::MAX - 1, Ordering::Relaxed);
    let axes = [
        ProviderNdgridAxis { handle: &x },
        ProviderNdgridAxis { handle: &y },
    ];
    assert!(provider
        .ndgrid(&ProviderNdgridRequest {
            axes: &axes,
            output_shape: &[2, 2],
            output_count: 2
        })
        .is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), 2);
    assert_eq!(
        futures::executor::block_on(provider.download(&x))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
    assert_eq!(
        futures::executor::block_on(provider.download(&y))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
    drop((x, y));
    assert_eq!(provider.allocations.record_count().unwrap(), 0);
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn unpolled_and_polled_future_drop_release_owned_handles_not_borrowed_inputs() {
    use std::future::Future;
    use std::task::{Context, Poll};
    async fn hold(handle: GpuTensorHandle) {
        futures::future::pending::<()>().await;
        drop(handle);
    }
    let provider = provider();
    let borrowed = upload(&provider);
    let future = hold(upload(&provider));
    assert_eq!(provider.allocations.record_count().unwrap(), 2);
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), 1);
    let mut future = Box::pin(hold(upload(&provider)));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), 1);
    drop(borrowed);
    assert_eq!(provider.allocations.record_count().unwrap(), 0);
}

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn logical_allocation_invariant_survives_numeric_aliases_and_casts() {
    use runmat_accelerate_api::HostLogicalTensorView;
    let provider = provider();
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    let numeric_view = logical
        .with_view(logical.view().clone().with_logical(false))
        .unwrap();
    let numeric = upload(&provider);
    let identity_cast =
        futures::executor::block_on(provider.floating_cast_exec(&numeric, provider.precision))
            .unwrap();
    assert!(
        identity_cast.shares_allocation_with(&numeric),
        "ordinary identity cast stays zero-copy"
    );
    assert!(provider
        .scatter_linear_exec(&numeric_view, &[0, 1], &numeric)
        .is_err());
    assert_eq!(
        futures::executor::block_on(provider.download(&logical))
            .unwrap()
            .data,
        vec![0.0, 1.0]
    );
    let values = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0],
            shape: &[1, 2],
        })
        .unwrap();
    provider
        .scatter_linear_exec(&numeric_view, &[0, 1], &values)
        .unwrap();
    assert_eq!(
        futures::executor::block_on(provider.download(&logical))
            .unwrap()
            .data,
        vec![1.0, 0.0]
    );
    let cast =
        futures::executor::block_on(provider.floating_cast_exec(&numeric_view, provider.precision))
            .unwrap();
    assert!(
        !cast.shares_allocation_with(&logical),
        "mutable numeric cast cannot invalidate logical aliases"
    );
    let copied = provider
        .scatter_linear_functional_exec(&numeric_view, &[0, 1], &values)
        .unwrap();
    assert!(provider
        .allocations
        .requires_logical_encoding(&copied)
        .unwrap());
    assert_eq!(
        futures::executor::block_on(provider.download(&copied))
            .unwrap()
            .data,
        vec![1.0, 0.0]
    );
    provider
        .scatter_linear_exec(&cast, &[0, 1], &numeric)
        .unwrap();
    assert_eq!(
        futures::executor::block_on(provider.download(&logical))
            .unwrap()
            .data,
        vec![1.0, 0.0]
    );
    assert_eq!(
        futures::executor::block_on(provider.download(&cast))
            .unwrap()
            .data,
        vec![1.0, 2.0]
    );
}
