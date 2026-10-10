use super::*;

#[test]
fn raw_upload_unsupported_only_falls_back_before_buffer_creation() {
    let before = numeric_upload_failure(
        runmat_accelerate_api::unsupported_provider_operation("pre-effect class"),
        false,
    );
    assert!(before.is_unsupported());

    let after = numeric_upload_failure(
        runmat_accelerate_api::unsupported_provider_operation("late provider failure"),
        true,
    );
    assert!(!after.is_unsupported());
    let mut source: &(dyn std::error::Error + 'static) = &after;
    while !source.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
        source = source.source().expect("retained unsupported source");
    }

    // The same classification is used after the observed completion fence:
    // a late transport failure keeps its source and cannot authorize replay.
    let completion = numeric_upload_failure(
        anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "observed upload submission failed",
        )),
        true,
    );
    assert!(!completion.is_unsupported());
    let source = std::error::Error::source(&completion).expect("original completion cause");
    assert_eq!(
        source
            .downcast_ref::<std::io::Error>()
            .expect("transport cause")
            .kind(),
        std::io::ErrorKind::BrokenPipe
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires a live WGPU adapter; run explicitly for post-buffer numeric upload qualification"]
fn actual_post_buffer_failure_keeps_cause_and_retires_unpublished_output() {
    use runmat_accelerate_api::{
        upload_numeric_protecting, HostNumericDataView, HostNumericTensorView, NumericUploadError,
        UnsupportedProviderOperation,
    };
    use std::sync::atomic::Ordering;

    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter required");
    let singles = [7.0_f32];
    let doubles = [7.0_f64];
    let input_data = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => HostNumericDataView::F32(&singles),
        runmat_accelerate_api::ProviderPrecision::F64 => HostNumericDataView::F64(&doubles),
    };
    let view = HostNumericTensorView {
        data: input_data,
        shape: &[1, 1],
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
    };
    let baseline_retained = provider.allocations.retained_bytes();
    let input = upload_numeric_protecting(&provider, &view, &[]).expect("live borrowed input");
    let weak_input = input.downgrade();
    let retained_before = provider.allocations.retained_bytes();
    assert!(
        retained_before > baseline_retained,
        "input owns resident bytes"
    );
    let attempts_before = provider
        .numeric_upload_post_buffer_count
        .load(Ordering::SeqCst);
    let unsupported = runmat_accelerate_api::unsupported_provider_operation(
        "injected failure after actual upload buffer creation",
    )
    .downcast::<UnsupportedProviderOperation>()
    .expect("canonical unsupported marker");
    *provider.numeric_upload_post_buffer_fault.lock().unwrap() = Some(anyhow::Error::new(
        std::io::Error::new(std::io::ErrorKind::BrokenPipe, unsupported),
    ));

    let failure = upload_numeric_protecting(&provider, &view, &[&input]).unwrap_err();
    let NumericUploadError::Provider(cause) = failure else {
        panic!("typed provider failure")
    };
    assert!(
        !cause.is_unsupported(),
        "a created buffer forbids host replay"
    );
    let mut chain: &(dyn std::error::Error + 'static) = &cause;
    let mut broken_pipe = false;
    let mut unsupported = false;
    loop {
        if let Some(error) = chain.downcast_ref::<std::io::Error>() {
            broken_pipe |= error.kind() == std::io::ErrorKind::BrokenPipe;
            // io::Error retains its custom payload through get_ref(); source()
            // delegates to that payload's source rather than yielding it.
            unsupported |= error
                .get_ref()
                .is_some_and(|source| source.is::<UnsupportedProviderOperation>());
        }
        unsupported |= chain.is::<UnsupportedProviderOperation>();
        let Some(next) = chain.source() else { break };
        chain = next;
    }
    assert!(
        broken_pipe && unsupported,
        "original nested cause remains intact"
    );
    assert_eq!(
        provider
            .numeric_upload_post_buffer_count
            .load(Ordering::SeqCst),
        attempts_before + 1,
    );
    assert!(provider
        .numeric_upload_post_buffer_fault
        .lock()
        .unwrap()
        .is_none());
    assert_eq!(provider.allocations.retained_bytes(), retained_before);
    assert!(weak_input.acquire().is_ok(), "borrowed input remains live");
    input
        .release()
        .expect("retire borrowed input after failure");
    assert!(weak_input.acquire().is_err());
    assert_eq!(provider.allocations.retained_bytes(), baseline_retained);
}
