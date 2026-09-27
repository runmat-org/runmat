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
        with_provider_observation, ProviderObservationFailure, ProviderObservationStatus,
        ProviderObservationSupport,
    };
    use runmat_accelerate_api::{
        AccelContextHandle, AccelContextKind, AccelProvider, ProviderElementType,
        ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
        ProviderOperationIdentity, ProviderRepresentation, ProviderResidency, ProviderStorage,
    };
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("required WGPU device");
    let AccelContextHandle::Wgpu(context) =
        provider.export_context(AccelContextKind::Plotting).unwrap();
    let emit_empty = || {
        let mut pending = PendingSubmissions::new(context.access.clone());
        context
            .access
            .with_shared(|emission| {
                pending.submit(
                    emission,
                    emission
                        .device()
                        .create_command_encoder(&Default::default()),
                )
            })
            .expect("emission available")
            .expect("same device");
        pending.commit();
    };
    let value = ProviderRepresentation {
        element_type: ProviderElementType::F32,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: vec![1, 1],
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new("fusion.elementwise"),
        family: ProviderOperationFamily::Fusion,
        inputs: vec![value.clone()],
        outputs: vec![value],
        workload: Default::default(),
    };
    let begin = || match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt,
        ProviderObservationSupport::Unsupported => panic!("required native completion driver"),
    };
    let mut first = begin();
    with_provider_observation(Some(first.recorder()), || {
        emit_empty();
    });
    first.fail(ProviderObservationFailure::Cancelled);
    // Unrelated work uses the same access authority but no observation helper;
    // measurement cannot depend on the newest global queue prefix.
    context
        .access
        .with_shared(|emission| {
            emission
                .queue()
                .submit(Some(emission.encode("unrelated", |_| {})));
        })
        .unwrap();
    assert_eq!(
        first.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    let mut next = begin();
    with_provider_observation(Some(next.recorder()), || {
        emit_empty();
    });
    next.publish().unwrap();
    // No second caller poll/submit is needed to drive native completion.
    struct Signal(std::sync::mpsc::SyncSender<()>);
    impl std::task::Wake for Signal {
        fn wake(self: Arc<Self>) {
            let _ = self.0.try_send(());
        }
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let waker = std::task::Waker::from(Arc::new(Signal(sender)));
    let mut cx = std::task::Context::from_waker(&waker);
    let evidence = match next.poll_completed(&mut cx) {
        std::task::Poll::Ready(Some(evidence)) => evidence,
        std::task::Poll::Pending => {
            receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("native progress must wake without caller poll");
            match next.poll_completed(&mut cx) {
                std::task::Poll::Ready(Some(evidence)) => evidence,
                _ => panic!("completion wake must carry evidence"),
            }
        }
        _ => panic!("undelivered receipt"),
    };
    assert!(matches!(
        evidence.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}
