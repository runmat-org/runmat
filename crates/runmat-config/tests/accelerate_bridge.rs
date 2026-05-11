#![cfg(feature = "accelerate")]

use runmat_accelerate::{
    AccelPowerPreference as RuntimePowerPreference, AccelerateInitOptions,
    AccelerateProviderPreference as RuntimeProviderPreference,
    AutoOffloadLogLevel as RuntimeAutoLogLevel,
};
use runmat_config::{
    AccelPowerPreference, AccelerateConfig, AccelerateProviderPreference, AutoOffloadConfig,
    AutoOffloadLogLevel,
};

#[test]
fn accelerate_config_converts_to_runtime_options() {
    let config = AccelerateConfig {
        enabled: false,
        provider: AccelerateProviderPreference::InProcess,
        allow_inprocess_fallback: false,
        wgpu_power_preference: AccelPowerPreference::HighPerformance,
        wgpu_force_fallback_adapter: true,
        auto_offload: AutoOffloadConfig {
            enabled: false,
            calibrate: false,
            log_level: AutoOffloadLogLevel::Info,
            ..Default::default()
        },
    };

    let options = AccelerateInitOptions::from(&config);

    assert!(!options.enabled);
    assert_eq!(options.provider, RuntimeProviderPreference::InProcess);
    assert!(!options.allow_inprocess_fallback);
    assert_eq!(
        options.wgpu_power_preference,
        RuntimePowerPreference::HighPerformance
    );
    assert!(options.wgpu_force_fallback_adapter);
    assert!(!options.auto_offload.enabled);
    assert!(!options.auto_offload.calibrate);
    assert_eq!(options.auto_offload.log_level, RuntimeAutoLogLevel::Info);
}
