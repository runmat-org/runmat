mod support;

use runmat_config::{ConfigLoader, RunMatConfig, TelemetryDrainMode};
use tempfile::TempDir;

const ENV_VARS: &[&str] = &[
    "RUNMAT_CONFIG",
    "RUNMAT_TELEMETRY_ENDPOINT",
    "RUNMAT_TELEMETRY_UDP_ENDPOINT",
    "RUNMAT_TELEMETRY_SYNC",
    "RUNMAT_TELEMETRY_DRAIN",
    "RUNMAT_TELEMETRY_DRAIN_TIMEOUT_MS",
];

#[test]
fn telemetry_env_overrides_respect_empty_values() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join(".runmat.yaml");
    ConfigLoader::save_to_file(&RunMatConfig::default(), &config_path).unwrap();

    std::env::set_var("RUNMAT_CONFIG", &config_path);
    std::env::set_var("RUNMAT_TELEMETRY_ENDPOINT", "https://custom.example/ingest");
    std::env::set_var("RUNMAT_TELEMETRY_UDP_ENDPOINT", "off");

    let config = ConfigLoader::load().unwrap();
    assert_eq!(
        config.telemetry.http_endpoint.as_deref(),
        Some("https://custom.example/ingest")
    );
    assert!(config.telemetry.udp_endpoint.is_none());

    support::clear_env(ENV_VARS);
}

#[test]
fn telemetry_runtime_env_overrides_promote_into_config() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join(".runmat.yaml");
    ConfigLoader::save_to_file(&RunMatConfig::default(), &config_path).unwrap();

    std::env::set_var("RUNMAT_CONFIG", &config_path);
    std::env::set_var("RUNMAT_TELEMETRY_SYNC", "1");
    std::env::set_var("RUNMAT_TELEMETRY_DRAIN", "none");
    std::env::set_var("RUNMAT_TELEMETRY_DRAIN_TIMEOUT_MS", "250");

    let config = ConfigLoader::load().unwrap();
    assert!(config.telemetry.sync_mode);
    assert_eq!(config.telemetry.drain_mode, TelemetryDrainMode::None);
    assert_eq!(config.telemetry.drain_timeout_ms, 250);

    support::clear_env(ENV_VARS);
}
