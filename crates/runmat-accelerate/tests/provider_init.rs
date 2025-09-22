#[test]
fn initialize_provider_registers() {
    let options = runmat_accelerate::AccelerateInitOptions {
        enabled: true,
        provider: runmat_accelerate::AccelerateProviderPreference::InProcess,
        allow_inprocess_fallback: true,
        wgpu_power_preference: runmat_accelerate::AccelPowerPreference::Auto,
        wgpu_force_fallback_adapter: false,
    };

    runmat_accelerate::initialize_acceleration_provider_with(&options);
    assert!(runmat_accelerate_api::provider().is_some());
}
