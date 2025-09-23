#[test]
fn initialize_provider_registers() {
    let options = runmat_accelerate::AccelerateInitOptions {
        enabled: true,
        provider: runmat_accelerate::AccelerateProviderPreference::InProcess,
        ..Default::default()
    };

    runmat_accelerate::initialize_acceleration_provider_with(&options);
    assert!(runmat_accelerate_api::provider().is_some());
}
