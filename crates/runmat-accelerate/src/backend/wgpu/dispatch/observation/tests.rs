use super::*;

#[test]
fn fence_size_is_mapping_aligned_and_small_not_payload_dependent() {
    assert_eq!(FENCE_REQUESTED_BYTES, 8);
    assert_eq!(FENCE_REQUESTED_BYTES % wgpu::MAP_ALIGNMENT, 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual WGPU device; run explicitly, never an unavailable-device pass"]
fn actual_private_fences_survive_cancel_and_unrelated_queue_submission() {
    use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationBudget, ProviderObservationFailure,
        ProviderObservationStatus, ProviderOperationReceipt,
    };
    use runmat_accelerate_api::{AccelContextHandle, AccelContextKind, AccelProvider};
    use std::num::NonZeroU32;
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU device");
    let AccelContextHandle::Wgpu(context) =
        provider.export_context(AccelContextKind::Plotting).unwrap()
    else {
        panic!("required WGPU context");
    };
    let budget = Arc::new(ProviderObservationBudget::new(NonZeroU32::new(1).unwrap()));
    let mut first = ProviderOperationReceipt::with_budget(budget.clone());
    with_provider_observation(Some(first.recorder()), || {
        submit(
            &context.device,
            &context.queue,
            context.device.create_command_encoder(&Default::default()),
        );
    });
    first.fail(ProviderObservationFailure::Cancelled);
    // Actual exported queue bypasses helper: observation cannot depend on a
    // callback attached to the newest global queue prefix.
    context.queue.submit(Some(
        context
            .device
            .create_command_encoder(&Default::default())
            .finish(),
    ));
    context.device.poll(wgpu::Maintain::Wait);
    assert_eq!(
        first.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    let mut next = ProviderOperationReceipt::with_budget(budget);
    with_provider_observation(Some(next.recorder()), || {
        submit(
            &context.device,
            &context.queue,
            context.device.create_command_encoder(&Default::default()),
        );
    });
    next.publish().unwrap();
    context.device.poll(wgpu::Maintain::Wait);
    assert!(matches!(
        next.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}
