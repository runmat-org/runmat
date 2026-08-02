use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("test-command")
}

fn run(arguments: &[&str], report_dir: &Path) -> Output {
    let workspace = tempfile::tempdir().expect("temporary test workspace");
    copy_tree(&fixture(), workspace.path());
    let cache = workspace.path().join(".runmat-package-cache");
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(workspace.path())
        .env("RUNMAT_PACKAGE_CACHE_DIR", cache)
        .args([
            "--color",
            "never",
            "--no-jit",
            "--plot-mode",
            "headless",
            "test",
        ])
        .args(arguments)
        .arg("--report-dir")
        .arg(report_dir)
        .output()
        .expect("runmat test command")
}

fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).expect("create fixture directory");
    for entry in std::fs::read_dir(source).expect("read fixture directory") {
        let entry = entry.expect("fixture entry");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("fixture metadata").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy fixture file");
        }
    }
}

#[test]
fn list_uses_static_discovery_without_starting_workers() {
    let temporary = tempfile::tempdir().unwrap();
    let output = run(&["--list", "--name", "testPasses"], temporary.path());
    assert!(output.status.success(), "{output:#?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("testPasses"), "{stdout}");
    assert!(!stdout.contains("testFails"), "{stdout}");
}

#[test]
fn process_worker_returns_stable_success_and_failure_exits() {
    let passing_reports = tempfile::tempdir().unwrap();
    let passing = run(
        &["--name", "testPasses", "--report", "json"],
        passing_reports.path(),
    );
    assert!(passing.status.success(), "{passing:#?}");
    assert!(find_named(passing_reports.path(), "test-results.json").is_some());

    let failing_reports = tempfile::tempdir().unwrap();
    let failing = run(&["--name", "testFails"], failing_reports.path());
    assert_eq!(failing.status.code(), Some(1), "{failing:#?}");
    assert!(
        String::from_utf8_lossy(&failing.stdout).contains("Failed"),
        "{failing:#?}"
    );
}

#[test]
fn explicit_native_session_and_none_modes_execute_without_claiming_process_isolation() {
    for isolation in ["session", "none"] {
        let reports = tempfile::tempdir().unwrap();
        let output = run(
            &[
                "--name",
                "testPasses",
                "--isolation",
                isolation,
                "--report",
                "json",
            ],
            reports.path(),
        );
        assert!(output.status.success(), "{isolation}: {output:#?}");
        assert!(
            find_named(reports.path(), "test-results.json").is_some(),
            "{isolation}: {output:#?}"
        );
    }
}

#[test]
fn process_workers_merge_backend_coverage_and_emit_all_formats() {
    let reports = tempfile::tempdir().unwrap();
    let output = run(
        &[
            "--name",
            "testPasses",
            "--coverage-format",
            "json",
            "--coverage-format",
            "lcov",
            "--coverage-format",
            "cobertura",
            "--coverage-format",
            "html",
        ],
        reports.path(),
    );
    assert!(output.status.success(), "{output:#?}");
    for name in [
        "coverage.json",
        "coverage.lcov",
        "coverage.xml",
        "coverage.html",
    ] {
        assert!(
            find_named(reports.path(), name).is_some(),
            "{name}: {output:#?}"
        );
    }
    let path = find_named(reports.path(), "coverage.json").unwrap();
    let coverage: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(
        coverage["sites"]
            .as_array()
            .is_some_and(|sites| !sites.is_empty()),
        "{coverage:#?}"
    );
}

#[test]
fn empty_selection_is_a_configuration_exit() {
    let temporary = tempfile::tempdir().unwrap();
    let output = run(&["--name", "does-not-exist"], temporary.path());
    assert_eq!(output.status.code(), Some(2), "{output:#?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no tests matched"),
        "{output:#?}"
    );
}

#[test]
fn timeout_is_a_test_failure_and_does_not_strand_the_worker() {
    let temporary = tempfile::tempdir().unwrap();
    let output = run(
        &["--name", "testHangs", "--timeout-ms", "25"],
        temporary.path(),
    );
    assert_eq!(output.status.code(), Some(1), "{output:#?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("TimedOut"),
        "{output:#?}"
    );
}

#[test]
fn hard_timeout_recovers_capacity_for_the_next_fixture_group_test() {
    let temporary = tempfile::tempdir().unwrap();
    let output = run(
        &["--name", "testRecovery", "--timeout-ms", "25"],
        temporary.path(),
    );
    assert_eq!(output.status.code(), Some(1), "{output:#?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("TimedOut"), "{output:#?}");
    assert!(stdout.contains("Passed"), "{output:#?}");
    assert!(stdout.contains("2 tests"), "{output:#?}");
}

fn find_named(root: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(root).ok()? {
        let path = entry.ok()?.path();
        if path.file_name().is_some_and(|value| value == name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_named(&path, name) {
                return Some(found);
            }
        }
    }
    None
}
