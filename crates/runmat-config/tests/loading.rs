use runmat_config::document::RunmatConfigFormat;
use runmat_config::runtime::{
    ConfigLoadOptions, ConfigSource, UserSettingsDocument, UserSettingsSource,
};
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
}

#[test]
fn recursive_layers_preserve_field_sources_and_replace_arrays() {
    let options = ConfigLoadOptions {
        user: UserSettingsSource::Document {
            source: r#"{"runtime":{"callstack_limit":512,"jit":{"enabled":false,"threshold":33},"foreign":{"java":{"options":["-Xmx1g","-ea"]}},"logging":{"otel":{"signals":["logs"],"resource_attributes":{"deployment.environment":"dev","deployment":"local"}}}}}"#.into(),
            format: RunmatConfigFormat::Json,
        },
        project_document: Some((r#"
[runtime.jit]
threshold = 42
[runtime.foreign.java]
options = ["-Xmx2g"]
[runtime.logging.otel]
signals = ["traces"]
resource_attributes = { "service.namespace" = "lab", deployment = "cluster" }
"#.into(), RunmatConfigFormat::Toml)),
        ..Default::default()
    };
    let resolved = ConfigLoader::load_with_options(&options).unwrap();
    assert_eq!(resolved.config.runtime.callstack_limit, 512);
    assert!(!resolved.config.jit.enabled);
    assert_eq!(resolved.config.jit.threshold, 42);
    assert_eq!(resolved.config.foreign.java.options, ["-Xmx2g"]);
    assert_eq!(
        resolved.config.logging.otel.signals,
        [runmat_config::runtime::OtelSignal::Traces]
    );
    assert_eq!(resolved.config.logging.otel.resource_attributes.len(), 3);
    assert!(matches!(
        resolved.source_segments(&[
            "runtime",
            "logging",
            "otel",
            "resource_attributes",
            "deployment.environment"
        ]),
        Some(ConfigSource::User { .. })
    ));
    assert!(matches!(
        resolved.source_segments(&[
            "runtime",
            "logging",
            "otel",
            "resource_attributes",
            "deployment"
        ]),
        Some(ConfigSource::Project { .. })
    ));
    assert!(matches!(
        resolved.source("runtime.jit.enabled"),
        Some(ConfigSource::User { path: None })
    ));
    assert!(matches!(
        resolved.source("runtime.jit.threshold"),
        Some(ConfigSource::Project { .. })
    ));
    assert!(matches!(
        resolved.source("runtime.gc.threads"),
        Some(ConfigSource::BuiltIn)
    ));
    assert!(matches!(
        resolved.source("runtime.foreign.java.options"),
        Some(ConfigSource::Project { .. })
    ));
}

#[test]
fn host_user_path_and_cluster_worker_policy_are_explicit() {
    let directory = TempDir::new().unwrap();
    let user = directory.path().join("settings-v1.json");
    std::fs::write(
        &user,
        r#"{"runtime":{"callstack_limit":512,"jit":{"enabled":false}}}"#,
    )
    .unwrap();
    std::fs::write(
        directory.path().join("runmat.toml"),
        "[runtime.jit]\nthreshold = 71\n",
    )
    .unwrap();
    let mut options = ConfigLoadOptions {
        user: UserSettingsSource::File(user.clone()),
        project_directory: Some(directory.path().into()),
        ..Default::default()
    };
    let local = ConfigLoader::load_with_options(&options).unwrap();
    assert_eq!(local.config.runtime.callstack_limit, 512);
    assert_eq!(local.config.jit.threshold, 71);
    assert_eq!(
        local.source("runtime.callstack_limit"),
        Some(&ConfigSource::User { path: Some(user) })
    );
    options.policy = runmat_config::runtime::ConfigurationPolicy::ClusterWorker;
    let worker = ConfigLoader::load_with_options(&options).unwrap();
    assert_eq!(
        worker.config.runtime.callstack_limit,
        RunMatRuntimeConfig::default().runtime.callstack_limit
    );
    assert!(worker.config.jit.enabled);
    assert_eq!(worker.config.jit.threshold, 71);
    let policy = runmat_config::runtime::ClusterRuntimePolicy::from_project_file(
        &directory.path().join("runmat.toml"),
    )
    .unwrap();
    let portable = serde_json::to_value(&policy).unwrap();
    assert!(portable.get("logging").is_none());
    assert!(portable.get("telemetry").is_none());
    assert_eq!(portable["schema_version"], 1);
    let cluster_config = policy.runtime_config().unwrap();
    assert_eq!(
        cluster_config.runtime.callstack_limit,
        RunMatRuntimeConfig::default().runtime.callstack_limit
    );
    assert!(!cluster_config.logging.otel.enabled);
    assert!(!cluster_config.telemetry.enabled);
}

#[test]
fn user_edits_keep_unknown_fields_comments_and_do_not_materialize_defaults() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "# settings header\n[future]\nenabled = true\n[runtime.jit]\n# compiler choice\nenabled = false # retain me\nfuture_knob = 17\n[runtime.logging]\nlevel = 'info'\n").unwrap();
    let mut document = UserSettingsDocument::load(&path).unwrap();
    document.edit(|runtime| runtime.jit.enabled = true).unwrap();
    document.set(&["gc", "threads"], 4).unwrap();
    document.save(&path).unwrap();
    let saved = std::fs::read_to_string(&path).unwrap();
    for preserved in [
        "# settings header",
        "# compiler choice",
        "# retain me",
        "future_knob = 17",
        "[future]",
        "level = 'info'",
    ] {
        assert!(saved.contains(preserved), "missing {preserved}: {saved}");
    }
    assert!(!saved.contains("callstack_limit"));
    assert_eq!(document.runtime().unwrap().gc.threads, Some(4));
    document.reset(&["jit", "enabled"]).unwrap();
    assert!(document.source().contains("future_knob = 17"));
    assert!(document.runtime().unwrap().jit.enabled);
    assert!(document.set(&["jit", "typo"], true).is_err());
    std::fs::write(&path, "# edited elsewhere\n").unwrap();
    assert!(document
        .save(&path)
        .unwrap_err()
        .to_string()
        .contains("changed on disk"));
}

#[test]
fn json_and_inline_toml_user_edits_validate_without_losing_siblings() {
    for (source, format) in [
        (
            r#"{"desktop":{"theme":"dark"},"runtime":{"jit":{"enabled":false,"future":42}}}"#,
            RunmatConfigFormat::Json,
        ),
        (
            "runtime = { jit = { enabled = false, future = 42 } }\n",
            RunmatConfigFormat::Toml,
        ),
    ] {
        let mut document = UserSettingsDocument::parse(source, format).unwrap();
        document.edit(|runtime| runtime.jit.enabled = true).unwrap();
        document.set(&["logging", "otel", "enabled"], true).unwrap();
        assert!(document.source().contains("future"));
        assert!(document.runtime().unwrap().logging.otel.enabled);
        let before = document.source().to_owned();
        assert!(document
            .set(&["foreign", "native", "timeout_ms"], 0)
            .is_err());
        assert_eq!(document.source(), before);
    }
}

#[test]
fn json_null_override_is_distinct_from_resetting_to_inherited_default() {
    let mut document = UserSettingsDocument::parse("{}", RunmatConfigFormat::Json).unwrap();
    document
        .edit(|runtime| runtime.plotting.gui = None)
        .unwrap();
    assert!(document.runtime().unwrap().plotting.gui.is_none());
    document.reset(&["plotting", "gui"]).unwrap();
    assert!(document.runtime().unwrap().plotting.gui.is_some());
    document
        .set(
            &["foreign", "python", "exact_version"],
            runmat_config::runtime::PythonExactVersion {
                major: 3,
                minor: 13,
                patch: 1,
            },
        )
        .unwrap();
    document
        .set(&["foreign", "python", "exact_version", "patch"], 2)
        .unwrap();
    assert_eq!(
        document
            .runtime()
            .unwrap()
            .foreign
            .python
            .exact_version
            .unwrap()
            .patch,
        2
    );
}
