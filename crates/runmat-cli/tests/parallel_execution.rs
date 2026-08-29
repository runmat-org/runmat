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

#[test]
fn spmd_executes_as_one_typed_multi_process_gang() {
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
    let script = workspace.path().join("spmd_results.m");
    fs::write(
        &script,
        r#"
pool = parpool(3);
wide_input = 0x0020000000000001u64;
spmd
  rank = spmdIndex();
  count = spmdSize();
  destination = mod(rank, count) + 1;
  source = mod(rank + count - 2, count) + 1;
  exchanged = spmdSendReceive(destination, source, uint16(rank));
  joined = spmdCat(uint16(rank), 2);
  total = spmdReduce(@plus, uint32(rank));
  wide_output = wide_input;
  if rank == 2
    assigned_on_second = uint16(22);
  end
end
assert(wide_output{3} == wide_input);
fprintf("SPMD_RANKS %.0f %.0f %.0f\n", rank{:});
fprintf("SPMD_COUNT %.0f %.0f %.0f\n", count{[1, 2, 3]});
fprintf("SPMD_EXCHANGE %.0f %.0f %.0f\n", exchanged{1}, exchanged{2}, exchanged{3});
fprintf("SPMD_JOINED %.0f %.0f %.0f\n", joined{1});
fprintf("SPMD_TOTAL %.0f\n", total{1});
fprintf("SPMD_OPTIONAL %.0f\n", assigned_on_second{2});
fprintf("SPMD_WIDE_OK\n");
"#,
    )
    .expect("write SPMD script");

    let output = run_script(&script, &config);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "SPMD script failed. stdout: {stdout} stderr: {stderr}"
    );
    for expected in [
        "SPMD_RANKS 1 2 3",
        "SPMD_COUNT 3 3 3",
        "SPMD_EXCHANGE 3 1 2",
        "SPMD_JOINED 1 2 3",
        "SPMD_TOTAL 6",
        "SPMD_OPTIONAL 22",
        "SPMD_WIDE_OK",
    ] {
        assert!(
            stdout.contains(expected),
            "missing {expected:?}. stdout: {stdout} stderr: {stderr}"
        );
    }

    let missing_script = workspace.path().join("spmd_unassigned.m");
    fs::write(
        &missing_script,
        r#"
pool = parpool(3);
spmd
  rank = spmdIndex();
  if rank == 2
    assigned_on_second = uint16(22);
  end
end
missing = assigned_on_second{1};
"#,
    )
    .expect("write unassigned-output script");
    let missing = run_script(&missing_script, &config);
    let missing_stdout = String::from_utf8_lossy(&missing.stdout);
    let missing_stderr = String::from_utf8_lossy(&missing.stderr);
    assert!(
        !missing.status.success(),
        "unassigned Composite entry unexpectedly became a value. stdout: {missing_stdout} stderr: {missing_stderr}"
    );
    assert!(
        missing_stderr.contains("RunMat:CompositeEntryUnavailable"),
        "unassigned Composite entry lost its typed diagnostic. stdout: {missing_stdout} stderr: {missing_stderr}"
    );
}

#[test]
fn resizing_a_native_pool_retires_existing_distributed_values() {
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
    let script = workspace.path().join("stale_distributed_lease.m");
    fs::write(
        &script,
        r#"
pool = parpool(2);
values = distributed(uint64([1, 0x0020000000000001u64]));
parpool(1);
local = getLocalPart(values);
"#,
    )
    .expect("write stale lease script");

    let output = run_script(&script, &config);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "retired distributed lease unexpectedly remained readable. stdout: {stdout} stderr: {stderr}"
    );
    assert!(
        stderr.contains("RunMat:parallel:StaleDistributedLease")
            && stderr.contains("pool generation has been retired"),
        "retired distributed lease lost its typed diagnostic. stdout: {stdout} stderr: {stderr}"
    );
}
