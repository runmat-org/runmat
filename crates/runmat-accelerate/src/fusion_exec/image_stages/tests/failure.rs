use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn final_power_failure_keeps_both_published_moments_and_never_replays() {
    let provider = provider(Some(12));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let (mut session, input, input_probe) = begin(provider);
    let mean = futures::executor::block_on(session.mean()).expect("mean").0;
    let sigma = futures::executor::block_on(session.sigma())
        .expect("sigma")
        .0;
    let error = futures::executor::block_on(session.output()).expect_err("final power failure");
    assert!(!error.is_unsupported(), "published moments forbid replay");
    assert!(futures::executor::block_on(session.output()).is_err());
    assert_eq!(provider.calls.load(Ordering::SeqCst), 13, "no hook replay");
    let probes = provider.probes();
    assert_eq!(probes.len(), 12, "failed power produced no result");
    assert_eq!(probes[1].retirements(), 0, "published mean remains live");
    assert_eq!(probes[8].retirements(), 0, "published sigma remains live");
    for scratch in [0, 2, 3, 4, 5, 6, 7, 9, 10, 11] {
        assert_eq!(probes[scratch].retirements(), 1);
    }
    drop(session);
    assert_eq!(input_probe.retirements(), 0, "borrowed input remains live");
    drop(mean);
    drop(sigma);
    assert_eq!(probes[1].retirements(), 1);
    assert_eq!(probes[8].retirements(), 1);
    drop(input);
    assert_eq!(input_probe.retirements(), 1);
}
