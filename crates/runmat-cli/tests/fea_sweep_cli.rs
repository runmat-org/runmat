#![cfg(feature = "occt-native")]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use serde_yaml::{Mapping, Value};
use tempfile::TempDir;

fn key(value: &str) -> Value {
    Value::String(value.into())
}

fn mapping(value: &mut Value) -> &mut Mapping {
    let Value::Mapping(mapping) = value else {
        panic!("expected a YAML mapping")
    };
    mapping
}

fn set_string(parent: &mut Value, field: &str, value: impl Into<String>) {
    mapping(parent).insert(key(field), Value::String(value.into()));
}

fn exact_box_study(repository: &Path, study_id: &str) -> Value {
    let fixture = repository.join("crates/runmat-cli/tests/fixtures/box_static.fea");
    let mut study: Value = serde_yaml::from_str(
        &fs::read_to_string(&fixture)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", fixture.display())),
    )
    .expect("box study fixture must parse");
    mapping(&mut study).remove(key("kind"));
    set_string(&mut study, "id", study_id);
    let geometry = mapping(&mut study)
        .get_mut(key("geometry"))
        .expect("box study must contain geometry");
    set_string(
        geometry,
        "path",
        repository
            .join("crates/runmat-geometry/io/tests/fixtures/box.brep")
            .display()
            .to_string(),
    );
    study
}

fn session_contains_analysis_program(session: &Path) -> bool {
    fs::read_dir(session.join("artifacts"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| fs::read(entry.path()).ok())
        .filter_map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .any(|artifact| {
            artifact
                .pointer("/recipe/entrypoint")
                .and_then(serde_json::Value::as_str)
                == Some("analysis_workload")
        })
}

#[test]
fn exact_structural_sweep_uses_one_bounded_native_session() {
    let temporary = TempDir::new().expect("temporary sweep workspace");
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sweep_path = temporary.path().join("two-boxes.fea");
    let sweep = Value::Mapping(Mapping::from_iter([
        (key("version"), Value::Number(1.into())),
        (key("kind"), key("sweep")),
        (key("id"), key("two_box_sweep")),
        (key("fail_fast"), Value::Bool(true)),
        (
            key("studies"),
            Value::Sequence(vec![
                exact_box_study(&repository, "box_a"),
                exact_box_study(&repository, "box_b"),
            ]),
        ),
    ]));
    fs::write(
        &sweep_path,
        serde_yaml::to_string(&sweep).expect("sweep YAML must encode"),
    )
    .expect("sweep fixture must be written");
    let config_path = temporary.path().join("runmat.toml");
    fs::write(
        &config_path,
        "[runtime.accelerate]\nenabled = false\nprovider = \"inprocess\"\n",
    )
    .expect("test config must be written");

    let execution_state = temporary.path().join("execution-state");
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command.current_dir(&repository).args([
        "run",
        sweep_path.to_str().expect("UTF-8 sweep path"),
        "--json",
    ]);
    command
        .env("RUNMAT_CONFIG", config_path)
        .env(
            "RUNMAT_FEA_ARTIFACT_ROOT",
            temporary.path().join("artifacts"),
        )
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            temporary.path().join("study-artifacts"),
        )
        .env("RUNMAT_EXECUTION_STATE_DIR", &execution_state)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("runmat sweep process must launch");
    let mut maximum_live_sessions = 0;
    let mut observed_analysis_sessions = BTreeSet::new();
    loop {
        let live_sessions = fs::read_dir(execution_state.join("sessions"))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .inspect(|entry| {
                        if session_contains_analysis_program(&entry.path()) {
                            observed_analysis_sessions.insert(entry.file_name());
                        }
                    })
                    .count()
            })
            .unwrap_or(0);
        maximum_live_sessions = maximum_live_sessions.max(live_sessions);
        if child.try_wait().expect("sweep process status").is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .expect("runmat sweep process must complete");
    assert!(
        output.status.success(),
        "sweep failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("sweep output must be JSON");
    assert_eq!(payload["study_count"], 2);
    assert_eq!(payload["success_count"], 2);
    assert_eq!(payload["failed_count"], 0);
    assert_eq!(payload["run_entries"].as_array().map(Vec::len), Some(2));

    assert_eq!(
        maximum_live_sessions, 1,
        "the complete sweep must use exactly one live bounded native scheduler session"
    );
    assert_eq!(
        observed_analysis_sessions.len(),
        1,
        "the complete sweep must use one shared analysis scheduler session"
    );
}
