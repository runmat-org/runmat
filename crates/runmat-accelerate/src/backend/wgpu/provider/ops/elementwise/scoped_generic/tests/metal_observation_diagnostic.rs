//! Opt-in diagnostic: one unchanged unary operation, one observation case per
//! process. Run under an external watchdog; production progress waits remain.
use super::*;
use runmat_accelerate_api::observation::with_provider_observation;
use runmat_execution::profiling::{with_command_profiling, CommandProfilingRoot};
use std::future::{poll_fn, Future};

#[test]
#[ignore = "requires actual F32 Metal hardware; select one case and use an external watchdog"]
fn actual_metal_sin_observation_matrix() {
    let case = std::env::var("RUNMAT_TEST_METAL_OBSERVATION_CASE")
        .expect("select plain, direct-counts, direct-compute-timed, direct-timed, selected-fence, or selected-timed");
    let (command_active, provider_active, timestamps) = match case.as_str() {
        "plain" => (false, false, true),
        "direct-counts" => (true, false, false),
        "direct-compute-timed" => (true, false, true),
        "direct-timed" => (true, false, true),
        "selected-fence" => (false, true, false),
        "selected-timed" => (false, true, true),
        _ => panic!("unknown diagnostic case: {case}"),
    };
    let mark = |stage: &str| {
        eprintln!(
            "RM1064_METAL_DIAGNOSTIC case={case} pid={} stage={stage}",
            std::process::id()
        );
    };
    mark("provider-before");
    let mut provider = WgpuProvider::new(Default::default()).expect("required actual WGPU device");
    assert_eq!(
        provider.adapter_info.backend,
        wgpu::Backend::Metal,
        "requires Metal, not a fallback backend"
    );
    assert_eq!(provider.precision, NumericPrecision::F32);
    eprintln!("RM1064_METAL_DIAGNOSTIC case={case} adapter={:?} features={:?} scalar_workgroup={} timestamp_available={}",
        provider.adapter_info, provider.enabled_features, provider.workgroup_config.scalar,
        provider.timestamp_capability.is_some());
    if timestamps && (command_active || provider_active) {
        assert!(
            provider.timestamp_capability.is_some(),
            "timed case requires timestamp support"
        );
    }
    if !timestamps {
        // Test-only instrumentation suppression: do not change device features,
        // kernel source, launch geometry, progress owner, or submission waits.
        provider.timestamp_capability = None;
    }
    if case == "direct-compute-timed" {
        // Preserve timestamp period, enabled hardware features, and compute
        // queries; isolate only the command's additional encoder-wide pair.
        provider.timestamp_capability = provider
            .timestamp_capability
            .map(|capability| capability.without_encoder_timestamps_for_test());
    }
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this fixture owns the process-lifetime provider, like owner().
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    mark("provider-after");

    let data = [0.1_f32, 0.2, 0.3, 0.4];
    mark("upload-before");
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&data),
            shape: &[4, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("four-element F32 upload");
    mark("upload-after");

    let command = command_active.then(CommandProfilingRoot::new);
    let command_receipt = command.as_ref().map(CommandProfilingRoot::receipt);
    let mut observation = provider_active.then(|| receipt(provider));
    let recorder = observation.as_ref().map(ProviderOperationReceipt::recorder);
    mark("sin-before");
    let operation = provider.unary_sin(&input);
    let mut operation = std::pin::pin!(operation);
    let output = pollster::block_on(poll_fn(|cx| {
        with_command_profiling(command_receipt.clone(), || {
            with_provider_observation(recorder.clone(), || operation.as_mut().poll(cx))
        })
    }))
    .expect("identical unary_sin operation");
    mark("sin-after");

    // Download is deliberately outside both optional observation scopes in all
    // rows; this separates unary submission from the common readback barrier.
    mark("download-before");
    let downloaded =
        pollster::block_on(provider.download_numeric(&output)).expect("consume sin output");
    mark("download-after");
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("sin output must retain Single storage");
    };
    assert_eq!(actual.len(), data.len());
    for (&actual, input) in actual.iter().zip(data) {
        assert!(
            (actual - input.sin()).abs() < 2e-6,
            "sin({input}) = {actual}"
        );
    }
    eprintln!("RM1064_METAL_DIAGNOSTIC case={case} values={actual:?}");
    if let Some(observation) = &mut observation {
        mark("observation-before");
        observation.publish().expect("publish selected observation");
        let evidence = pollster::block_on(poll_fn(|cx| observation.poll_completed(cx)))
            .expect("selected observation completion evidence");
        eprintln!(
            "RM1064_METAL_DIAGNOSTIC case={case} observation_status={:?}",
            evidence.status
        );
        mark("observation-after");
    }
    if let Some(command) = command {
        eprintln!(
            "RM1064_METAL_DIAGNOSTIC case={case} command={:?}",
            command.seal()
        );
    }
    mark("complete");
}
