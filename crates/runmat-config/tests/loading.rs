use runmat_config::runtime::{ConfigLoader, FeaArtifactStoreMode, RunMatRuntimeConfig};
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
    assert!(format!("{err:#}").contains("verbosee"));
}

#[test]
fn runtime_loader_ignores_desktop_section() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[desktop.artifacts]
root = ".cache/artifacts"

[desktop.notebook]
on_error = "continue"

[runtime]
callstack_limit = 64
"#,
    )
    .unwrap();

    let runtime = ConfigLoader::load_from_file(&config_path).unwrap();
    assert_eq!(runtime.runtime.callstack_limit, 64);
}

#[test]
fn runtime_loader_migrates_legacy_acceleration_setting() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[desktop]
enable_gpu = false

[runtime]
callstack_limit = 64
"#,
    )
    .unwrap();

    let runtime = ConfigLoader::load_from_file(&config_path).unwrap();
    assert!(!runtime.accelerate.enabled);
    assert_eq!(runtime.runtime.callstack_limit, 64);
}

#[test]
fn runtime_loader_ignores_package_test_and_desktop_sections() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[test]
roots = ["tests"]
jobs = 4

[desktop]
artifact_root = ".artifacts"

[runtime]
callstack_limit = 96
"#,
    )
    .unwrap();

    let runtime = ConfigLoader::load_from_file(&config_path).unwrap();
    assert_eq!(runtime.runtime.callstack_limit, 96);
}

#[test]
fn runtime_fea_section_loads_artifact_config() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime.fea]
artifact_store = "filesystem"
artifact_root = ".runmat/fea"
artifact_max_runs = 12
artifact_max_runs_per_kind = 3
study_artifact_root = ".runmat/studies"
thermo_field_artifact_root = ".runmat/thermo-fields"
"#,
    )
    .unwrap();

    let runtime = ConfigLoader::load_from_file(&config_path).unwrap();
    assert_eq!(
        runtime.fea.artifact_store,
        Some(FeaArtifactStoreMode::Filesystem)
    );
    assert_eq!(
        runtime.fea.artifact_root.as_deref(),
        Some(std::path::Path::new(".runmat/fea"))
    );
    assert_eq!(runtime.fea.artifact_max_runs, Some(12));
    assert_eq!(runtime.fea.artifact_max_runs_per_kind, Some(3));
    assert_eq!(
        runtime.fea.study_artifact_root.as_deref(),
        Some(std::path::Path::new(".runmat/studies"))
    );
    assert_eq!(
        runtime.fea.thermo_field_artifact_root.as_deref(),
        Some(std::path::Path::new(".runmat/thermo-fields"))
    );
}
