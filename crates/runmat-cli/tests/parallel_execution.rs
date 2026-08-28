use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

fn runmat_binary() -> PathBuf {
    let mut path = std::env::current_exe().expect("test executable has a path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("runmat");
    path
}

fn run_script(script: &Path, config: &Path) -> Output {
    let state = config
        .parent()
        .expect("test configuration has a parent")
        .join("execution-state");
    Command::new(runmat_binary())
        .args(["--no-jit", "run"])
        .arg(script)
        .env("RUNMAT_CONFIG", config)
        .env("RUNMAT_EXECUTION_STATE_DIR", state)
        .env("NO_GUI", "1")
        .output()
        .expect("runmat starts")
}

#[test]
fn parfor_executes_in_process_workers_and_assembles_results() {
    let workspace = TempDir::new().expect("temporary workspace");
    let config = workspace.path().join("runmat.toml");
    fs::write(
        &config,
        r#"
[runtime.accelerate]
enabled = false
provider = "inprocess"
"#,
    )
    .expect("write test configuration");
    let script = workspace.path().join("parallel_results.m");
    fs::write(
        &script,
        r#"
values = zeros(1, 8);
task_seen = zeros(1, 8);
worker_seen = zeros(1, 8);
total = 0;
pool = parpool(2);
parfor (index = 1:8, 2)
  values(index) = index * 2;
  task = getCurrentTask();
  worker = getCurrentWorker();
  task_seen(index) = task.ID == task.ID;
  worker_seen(index) = worker.ID == worker.ID;
  total = total + index;
end
driver_job_empty = isempty(getCurrentJob());
offset = 1;
shifted = zeros(1, 9);
parfor index = 1:8
  shifted(index + offset) = index;
end
serial = zeros(1, 4);
parfor (index = 1:4, 0)
  serial(index) = index;
end
rng(41);
parallel_random = zeros(1, 8);
parfor (index = 1:8, 2)
  parallel_random(index) = rand();
end
rng(41);
serial_random = zeros(1, 8);
parfor (index = 1:8, 0)
  serial_random(index) = rand();
end
initial_workers = pool.NumWorkers;
resized = parpool(1);
fprintf("PARFOR_VALUES %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", values);
fprintf("PARFOR_TOTAL %.0f\n", total);
fprintf("PARFOR_CONTEXT %.0f %.0f %.0f\n", sum(task_seen), sum(worker_seen), driver_job_empty);
fprintf("PARFOR_POOL %.0f\n", initial_workers);
fprintf("PARFOR_RESIZED_POOL %.0f\n", resized.NumWorkers);
fprintf("PARFOR_SHIFTED %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", shifted);
fprintf("PARFOR_SERIAL %.0f %.0f %.0f %.0f\n", serial);
fprintf("PARFOR_RANDOM %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", parallel_random);
fprintf("SERIAL_RANDOM %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", serial_random);
"#,
    )
    .expect("write parallel script");

    let output = run_script(&script, &config);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "parallel script failed. stdout: {stdout} stderr: {stderr}"
    );
    assert!(
        stdout.contains("PARFOR_VALUES 2 4 6 8 10 12 14 16"),
        "unexpected parallel slice output: {stdout}"
    );
    assert!(
        stdout.contains("PARFOR_TOTAL 36"),
        "unexpected parallel reduction output: {stdout}"
    );
    assert!(
        stdout.contains("PARFOR_CONTEXT 8 8 1"),
        "unexpected worker execution context. stdout: {stdout} stderr: {stderr}"
    );
    assert!(
        stdout.contains("PARFOR_POOL 2"),
        "unexpected explicit pool size: {stdout}"
    );
    assert!(
        stdout.contains("PARFOR_RESIZED_POOL 1"),
        "unexpected resized pool size: {stdout}"
    );
    assert!(
        stdout.contains("PARFOR_SHIFTED 0 1 2 3 4 5 6 7 8"),
        "unexpected affine slice output: {stdout}"
    );
    assert!(
        stdout.contains("PARFOR_SERIAL 1 2 3 4"),
        "unexpected zero-worker serial output: {stdout}"
    );
    let parallel_random = stdout
        .lines()
        .find_map(|line| line.strip_prefix("PARFOR_RANDOM "))
        .expect("parallel random output");
    let serial_random = stdout
        .lines()
        .find_map(|line| line.strip_prefix("SERIAL_RANDOM "))
        .expect("serial random output");
    assert_eq!(parallel_random, serial_random);
}

#[test]
fn parfor_preserves_structured_worker_failures_and_source_locations() {
    let workspace = TempDir::new().expect("temporary workspace");
    let config = workspace.path().join("runmat.toml");
    fs::write(
        &config,
        r#"
[runtime.accelerate]
enabled = false
provider = "inprocess"
"#,
    )
    .expect("write test configuration");
    let script = workspace.path().join("parallel_failure.m");
    fs::write(
        &script,
        r#"pool = parpool(2);
input = ones(1, 4);
values = zeros(1, 4);
parfor (index = 1:4, 2)
  values(index) = input(5);
end
"#,
    )
    .expect("write parallel failure script");

    let output = run_script(&script, &config);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "parallel failure unexpectedly succeeded. stdout: {stdout} stderr: {stderr}"
    );
    assert!(
        stderr.contains("RunMat:IndexOutOfBounds"),
        "parallel failure lost its identifier. stdout: {stdout} stderr: {stderr}"
    );
    assert!(
        stderr.contains("parallel_failure.m:5"),
        "parallel failure lost its worker source location. stdout: {stdout} stderr: {stderr}"
    );
}
