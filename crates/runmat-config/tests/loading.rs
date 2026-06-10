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
    assert!(err.to_string().contains("Failed to parse TOML config"));
}

#[test]
fn runtime_loader_ignores_desktop_section() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("runmat.toml");
    std::fs::write(
        &config_path,
        r#"
[desktop]
artifact_root = ".cache/artifacts"
notebook_run_mode = "continue_on_error"

[runtime]
callstack_limit = 64
"#,
    )
    .unwrap();

    let runtime = ConfigLoader::load_from_file(&config_path).unwrap();
    assert_eq!(runtime.runtime.callstack_limit, 64);
}

#[test]
fn runtime_fea_section_loads_artifact_and_prep_config() {
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
geometry_prep_artifact_root = ".runmat/geometry-prep"
geometry_prep_max_artifacts = 8
geometry_prep_max_artifacts_per_geometry = 2
geometry_prep_max_age_seconds = 3600
geometry_prep_require_latest_revision = false
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
        runtime.fea.geometry_prep_artifact_root.as_deref(),
        Some(std::path::Path::new(".runmat/geometry-prep"))
    );
    assert_eq!(runtime.fea.geometry_prep_max_artifacts, Some(8));
    assert_eq!(
        runtime.fea.geometry_prep_max_artifacts_per_geometry,
        Some(2)
    );
    assert_eq!(runtime.fea.geometry_prep_max_age_seconds, Some(3600));
    assert_eq!(
        runtime.fea.geometry_prep_require_latest_revision,
        Some(false)
    );
    assert_eq!(
        runtime.fea.thermo_field_artifact_root.as_deref(),
        Some(std::path::Path::new(".runmat/thermo-fields"))
    );
}
