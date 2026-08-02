use runmat_config::desktop::{
    DesktopNotebookOnError, DesktopNotebookRerunAfterCancel, DesktopRunHistoryMode,
    DesktopRunLogMode,
};
use runmat_config::document::{
    migrate_legacy_desktop_config, migrate_legacy_desktop_config_between,
    migrate_legacy_desktop_config_into, RunmatConfigDocument, RunmatConfigFormat,
    RunmatConfigPatch,
};
use std::path::Path;

#[test]
fn parses_complete_toml_and_runtime_projection() {
    let document = RunmatConfigDocument::parse(
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[desktop.artifacts]
root = ".cache/runmat"

[desktop.run_history]
mode = "full"
trace = false
logs = "errors"

[desktop.script]
clear_workspace_before_run = false
clear_figures_before_run = true

[desktop.notebook]
on_error = "continue"
rerun_after_cancel = "all"

[runtime.accelerate]
enabled = false

[runtime.plotting.export]
scene_budget_bytes = 4194304
"#,
        RunmatConfigFormat::Toml,
    )
    .unwrap();
    assert_eq!(
        document.desktop().artifacts.root,
        Path::new(".cache/runmat")
    );
    assert_eq!(
        document.desktop().run_history.mode,
        DesktopRunHistoryMode::Full
    );
    assert_eq!(
        document.desktop().run_history.logs,
        DesktopRunLogMode::Errors
    );
    assert_eq!(
        document.desktop().notebook.on_error,
        DesktopNotebookOnError::Continue
    );
    assert_eq!(
        document.desktop().notebook.rerun_after_cancel,
        DesktopNotebookRerunAfterCancel::All
    );
    assert!(!document.runtime().accelerate.enabled);
    assert_eq!(
        document
            .runtime()
            .plotting
            .export
            .as_ref()
            .unwrap()
            .scene_budget_bytes,
        4 * 1024 * 1024
    );
}

#[test]
fn merges_toml_legacy_values_into_canonical_json() {
    let legacy = r#"[desktop]
artifact_root = .legacy
enable_gpu = false

[future]
answer = 42
"#;
    let destination = r#"{
  "package": { "name": "demo" },
  "desktop": { "artifacts": { "root": ".canonical" } }
}"#;
    let migration = migrate_legacy_desktop_config_between(
        legacy,
        RunmatConfigFormat::Toml,
        destination,
        RunmatConfigFormat::Json,
    )
    .unwrap();
    let document =
        RunmatConfigDocument::parse(migration.source.clone(), RunmatConfigFormat::Json).unwrap();
    assert_eq!(
        document.desktop().artifacts.root,
        std::path::Path::new(".canonical")
    );
    assert!(!document.runtime().accelerate.enabled);
    let json: serde_json::Value = serde_json::from_str(&migration.source).unwrap();
    assert_eq!(json["future"]["answer"], 42);
    assert_eq!(json["package"]["name"], "demo");
}

#[test]
fn merges_legacy_desktop_values_into_existing_project_document() {
    let legacy = r#"
[desktop]
artifact_root = custom-artifacts
enable_gpu = false
notebook_persistence_mode = full
show_internal_artifacts = true

[runtime]
verbose = true
"#;
    let existing = r#"# keep this comment
[package]
name = "demo"

[desktop.run_history]
mode = "off"
"#;
    let merged = migrate_legacy_desktop_config_into(legacy, existing, RunmatConfigFormat::Toml)
        .expect("merge legacy desktop config");
    assert!(merged.source.contains("# keep this comment"));
    assert!(merged.source.contains("name = \"demo\""));
    assert!(merged.source.contains("root = \"custom-artifacts\""));
    assert!(merged.source.contains("enabled = false"));
    assert!(merged.source.contains("mode = \"off\""));
    assert!(merged.source.contains("[runtime]\nverbose = true"));
    assert!(!merged.source.contains("show_internal_artifacts"));
}

#[test]
fn promotes_canonical_legacy_file_and_recursively_merges_json() {
    let canonical_legacy = r#"[desktop.artifacts]
root = "custom-artifacts"

[runtime]
verbose = true
"#;
    let promoted =
        migrate_legacy_desktop_config_into(canonical_legacy, "", RunmatConfigFormat::Toml)
            .expect("promote canonical legacy file");
    assert!(promoted.changed);
    assert!(promoted.removed_keys.is_empty());
    assert!(promoted.source.contains("root = \"custom-artifacts\""));
    assert!(promoted.source.contains("verbose = true"));

    let legacy_json = r#"{
  "desktop": {
    "artifact_root": "legacy-artifacts",
    "run_history": { "trace": false }
  },
  "runtime": { "verbose": true }
}"#;
    let destination_json = r#"{
  "desktop": {
    "artifacts": { "root": "canonical-artifacts" },
    "run_history": { "mode": "off" }
  }
}"#;
    let merged =
        migrate_legacy_desktop_config_into(legacy_json, destination_json, RunmatConfigFormat::Json)
            .expect("recursively merge JSON legacy file");
    let value: serde_json::Value = serde_json::from_str(&merged.source).unwrap();
    assert_eq!(value["desktop"]["artifacts"]["root"], "canonical-artifacts");
    assert_eq!(value["desktop"]["run_history"]["mode"], "off");
    assert_eq!(value["desktop"]["run_history"]["trace"], false);
    assert_eq!(value["runtime"]["verbose"], true);
}

#[test]
fn patches_toml_without_rewriting_unrelated_content() {
    let source = "# keep this comment\n[package]\nname = \"demo\"\n";
    let document =
        RunmatConfigDocument::parse(source, RunmatConfigFormat::Toml).expect("parse source");
    let mut patch = RunmatConfigPatch::default();
    patch.desktop.run_history.mode = Some(DesktopRunHistoryMode::Full);
    patch.runtime.accelerate_enabled = Some(false);
    let patched = document.patched(&patch).unwrap();
    assert!(patched.source().starts_with("# keep this comment\n"));
    assert!(patched.source().contains("[package]\nname = \"demo\""));
    assert!(patched
        .source()
        .contains("[desktop.run_history]\nmode = \"full\""));
    assert!(patched
        .source()
        .contains("[runtime.accelerate]\nenabled = false"));
}

#[test]
fn json_and_toml_resolve_equivalent_desktop_defaults() {
    let toml = RunmatConfigDocument::parse("", RunmatConfigFormat::Toml).unwrap();
    let json = RunmatConfigDocument::parse("{}", RunmatConfigFormat::Json).unwrap();
    assert_eq!(toml.desktop(), json.desktop());
}

#[test]
fn rejects_unknown_desktop_keys() {
    let error = RunmatConfigDocument::parse(
        "[desktop.run_history]\nmagic = true\n",
        RunmatConfigFormat::Toml,
    )
    .unwrap_err();
    assert!(error.to_string().contains("magic"));
}

#[test]
fn migrates_shipped_invalid_flat_toml_without_losing_other_sections() {
    let source = r#"# project comment
[package]
name = "demo"

[desktop]
artifact_root = .cache/runmat
enable_gpu = false
persist_agents = true
background_runtime_diagnosis = true
command_window_as_tab = false
show_internal_artifacts = true
notebook_persistence_mode = smart
notebook_interrupted_run_mode = full
notebook_run_mode = continue_on_error
notebook_auto_restore_workspace = true
run_clear_workspace_before_execution = false
run_clear_figures_before_execution = true
run_persist_trace = false
run_persist_logs_mode = errors
runtime_error_reporting = false
figure_scene_budget_bytes = 4194304

[future]
answer = 42
"#;
    let migration = migrate_legacy_desktop_config(source, RunmatConfigFormat::Toml).unwrap();
    assert!(migration.changed);
    assert_eq!(migration.removed_keys.len(), 16);
    assert!(migration.source.starts_with("# project comment\n"));
    assert!(migration.source.contains("[future]\nanswer = 42"));
    assert!(!migration.source.contains("persist_agents"));
    assert!(!migration.source.contains("background_runtime_diagnosis"));

    let document = RunmatConfigDocument::parse(migration.source, RunmatConfigFormat::Toml).unwrap();
    assert_eq!(
        document.desktop().artifacts.root,
        Path::new(".cache/runmat")
    );
    assert_eq!(
        document.desktop().run_history.mode,
        DesktopRunHistoryMode::Budgeted
    );
    assert_eq!(
        document.desktop().notebook.rerun_after_cancel,
        DesktopNotebookRerunAfterCancel::All
    );
    assert_eq!(
        document.desktop().notebook.on_error,
        DesktopNotebookOnError::Continue
    );
    assert!(!document.desktop().script.clear_workspace_before_run);
    assert!(!document.desktop().run_history.trace);
    assert_eq!(
        document.desktop().run_history.logs,
        DesktopRunLogMode::Errors
    );
    assert!(!document.runtime().accelerate.enabled);
    assert_eq!(
        document
            .runtime()
            .plotting
            .export
            .as_ref()
            .unwrap()
            .scene_budget_bytes,
        4 * 1024 * 1024
    );
}

#[test]
fn canonical_values_win_over_legacy_values_during_migration() {
    let source = r#"[desktop]
artifact_root = old-artifacts
enable_gpu = false

[desktop.artifacts]
root = "canonical-artifacts"

[runtime.accelerate]
enabled = true
"#;
    let migration = migrate_legacy_desktop_config(source, RunmatConfigFormat::Toml).unwrap();
    let document = RunmatConfigDocument::parse(migration.source, RunmatConfigFormat::Toml).unwrap();
    assert_eq!(
        document.desktop().artifacts.root,
        Path::new("canonical-artifacts")
    );
    assert!(document.runtime().accelerate.enabled);
}

#[test]
fn migrates_legacy_json_and_preserves_unknown_top_level_sections() {
    let source = r#"{
  "package": { "name": "demo" },
  "desktop": {
    "artifact_root": ".cache/runmat",
    "notebook_persistence_mode": "full",
    "show_internal_artifacts": true
  },
  "future": { "answer": 42 }
}"#;
    let migration = migrate_legacy_desktop_config(source, RunmatConfigFormat::Json).unwrap();
    assert!(migration.changed);
    let value: serde_json::Value = serde_json::from_str(&migration.source).unwrap();
    assert_eq!(value["future"]["answer"], 42);
    assert!(value["desktop"].get("show_internal_artifacts").is_none());
    let document = RunmatConfigDocument::parse(migration.source, RunmatConfigFormat::Json).unwrap();
    assert_eq!(
        document.desktop().run_history.mode,
        DesktopRunHistoryMode::Full
    );
}

#[test]
fn replacing_runtime_preserves_unowned_sections() {
    let source =
        "# keep\n[package]\nname = \"demo\"\n\n[desktop.artifacts]\nroot = \".artifacts\"\n";
    let document = RunmatConfigDocument::parse(source, RunmatConfigFormat::Toml).unwrap();
    let mut runtime = document.runtime().clone();
    runtime.runtime.callstack_limit = 777;
    let replaced = document.with_runtime(&runtime).unwrap();
    assert!(replaced.source().contains("# keep\n"));
    assert!(replaced.source().contains("[package]\nname = \"demo\""));
    assert!(replaced
        .source()
        .contains("[desktop.artifacts]\nroot = \".artifacts\""));
    assert_eq!(replaced.runtime().runtime.callstack_limit, 777);
}
