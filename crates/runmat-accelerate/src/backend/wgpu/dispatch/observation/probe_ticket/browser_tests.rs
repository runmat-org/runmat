use super::super::*;
use crate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::observation::{
    with_provider_observation, ProviderObservationStatus, ProviderObservationSupport,
};
use runmat_accelerate_api::{
    AccelContextHandle, AccelContextKind, AccelProvider, ProviderElementType,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test(async)]
async fn browser_callback_completes_optional_probe_ticket_after_cleanup() {
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual browser WebGPU adapter");
    let AccelContextHandle::Wgpu(context) =
        provider.export_context(AccelContextKind::Plotting).unwrap();
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
    let mut operation = match provider.begin_operation_observation(&query) {
        ProviderObservationSupport::Observed(receipt) => receipt,
        ProviderObservationSupport::Unsupported => panic!("browser observation required"),
    };
    let ticket = with_provider_observation(Some(operation.recorder()), || {
        let mut pending = PendingSubmissions::new(context.access.clone());
        let ticket = pending.arm_probe_completion();
        context
            .access
            .with_shared(|emission| {
                let encoder = pending.profiled_encoder(emission, &Default::default());
                pending.submit(emission, encoder)
            })
            .expect("browser emission")
            .expect("same device");
        pending.commit();
        ticket
    });
    assert_eq!(ticket.fence_requested_bytes(), Some(8));
    assert_eq!(ticket.state(), ProbeFenceState::Pending);
    operation.publish().unwrap();
    let evidence = futures::future::poll_fn(|cx| operation.poll_completed(cx))
        .await
        .expect("browser callback receipt");
    assert!(matches!(
        evidence.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(ticket.state(), ProbeFenceState::Completed);
}
