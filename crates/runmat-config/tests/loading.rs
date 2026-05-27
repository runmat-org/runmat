use runmat_config::runtime::{ConfigLoader, RunMatRuntimeConfig};
use tempfile::TempDir;

#[test]
fn file_loading() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");

    let mut config = RunMatRuntimeConfig::default();
    config.runtime.callstack_limit = 333;
    config.jit.threshold = 20;

    ConfigLoader::save_to_file(&config, &config_path).unwrap();
    let loaded = ConfigLoader::load_from_file(&config_path).unwrap();

    assert_eq!(loaded.runtime.callstack_limit, 333);
    assert_eq!(loaded.jit.threshold, 20);
}

#[test]
fn runtime_section_rejects_unknown_keys() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
callstack_limit = 64
verbosee = true
"#,
    )
    .unwrap();

    let err = ConfigLoader::load_from_file(&config_path)
        .expect_err("unknown runtime keys should fail validation");
    assert!(err.to_string().contains("Failed to parse TOML config"));
}
