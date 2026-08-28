#![cfg(not(target_family = "wasm"))]

use std::fs;
use std::process::Command;

use runmat_python::{discover_python, PythonDiscoveryRequest};

static ISOLATED_PYTHON_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn python_is_available() -> bool {
    discover_python(&PythonDiscoveryRequest::default()).is_ok()
}

fn run_script(source: &str) -> std::process::Output {
    let directory = tempfile::tempdir().expect("create isolated Python test directory");
    let script = directory.path().join("main.m");
    fs::write(&script, source).expect("write isolated Python script");
    let config = directory.path().join("runmat.toml");
    fs::write(
        &config,
        r#"
[runtime.accelerate]
enabled = false
provider = "inprocess"
"#,
    )
    .expect("write isolated Python configuration");
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .arg(script)
        .current_dir(directory.path())
        .env("RUNMAT_CONFIG", config)
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            directory.path().join("execution-state"),
        )
        .env("NO_GUI", "1")
        .output()
        .expect("run isolated Python script")
}

#[test]
fn same_binary_host_preserves_values_objects_callbacks_and_restart_fencing() {
    let _guard = ISOLATED_PYTHON_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !python_is_available() {
        eprintln!("skipping isolated Python test because CPython is unavailable");
        return;
    }
    let output = run_script(
        r#"
environment = pyenv("ExecutionMode", "OutOfProcess");
items = py.list({int32(10), int32(20)});
items(2) = int32(42);
mapped = py.list(py.map(@pythonCallback, {int64(40)}));
values = uint64([9007199254740993, 18446744073709551615]);
incremented = py.numpy.add(values, uint64([1, 0]));
disp(items(2));
disp(mapped(1));
disp(class(incremented));
terminate(environment);
try
    items.append(int32(3));
    disp("handle-was-not-fenced");
catch failure
    disp(failure.identifier);
end
replacement = py.list({int32(7)});
disp(replacement(1));

function y = pythonCallback(x)
    y = py.operator.add(x, int64(2));
end
"#,
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.split_whitespace().collect::<Vec<_>>(),
        ["42", "42", "uint64", "RunMat:Python:StaleHandle", "7"],
        "stdout={stdout}"
    );
}

#[test]
fn structured_python_exception_crosses_the_host_boundary() {
    let _guard = ISOLATED_PYTHON_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !python_is_available() {
        eprintln!("skipping isolated Python test because CPython is unavailable");
        return;
    }
    let output = run_script(
        r#"
pyenv("ExecutionMode", "OutOfProcess");
pyrun("raise ValueError('isolated failure')");
"#,
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ValueError"), "stderr={stderr}");
    assert!(stderr.contains("isolated failure"), "stderr={stderr}");
}
