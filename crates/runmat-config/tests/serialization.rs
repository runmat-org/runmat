use runmat_config::RunMatConfig;

#[test]
fn yaml_serialization() {
    let config = RunMatConfig::default();
    let yaml = serde_yaml::to_string(&config).unwrap();
    let parsed: RunMatConfig = serde_yaml::from_str(&yaml).unwrap();

    assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
    assert_eq!(parsed.jit.enabled, config.jit.enabled);
    assert_eq!(parsed.accelerate.provider, config.accelerate.provider);
}

#[test]
fn json_serialization() {
    let config = RunMatConfig::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    let parsed: RunMatConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.runtime.timeout, config.runtime.timeout);
    assert_eq!(parsed.plotting.mode, config.plotting.mode);
    assert_eq!(parsed.accelerate.enabled, config.accelerate.enabled);
}
