#[test]
fn initialize_provider_registers() {
    let options = runmat_accelerate::AccelerateInitOptions {
        enabled: true,
        provider: runmat_accelerate::AccelerateProviderPreference::InProcess,
        ..Default::default()
    };

    runmat_accelerate::initialize_acceleration_provider_with(&options);
    assert!(runmat_accelerate_api::provider().is_some());

    let disabled = runmat_accelerate::AccelerateInitOptions {
        enabled: false,
        allow_inprocess_fallback: false,
        ..Default::default()
    };
    runmat_accelerate::reinitialize_acceleration_provider_with(&disabled);
    assert!(runmat_accelerate_api::provider().is_none());
}
