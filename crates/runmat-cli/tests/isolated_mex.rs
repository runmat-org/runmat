#![cfg(not(target_family = "wasm"))]

use std::fs;
use std::process::Command;

use runmat_mex::MexBuild;

static SPINNING_HOST_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize_spinning_host_test() -> std::sync::MutexGuard<'static, ()> {
    SPINNING_HOST_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn c_string_path(path: &std::path::Path) -> String {
    path.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn run_script(
    directory: &std::path::Path,
    script: &str,
    mex_configuration: &str,
) -> std::process::Output {
    configured_command(directory, script, mex_configuration)
        .output()
        .unwrap()
}

fn configured_command(
    directory: &std::path::Path,
    script: &str,
    mex_configuration: &str,
) -> Command {
    let config = directory.join("runmat.toml");
    fs::write(
        &config,
        format!(
            r#"
[runtime.accelerate]
enabled = false
provider = "inprocess"
{mex_configuration}
"#
        ),
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command
        .arg(directory.join(script))
        .current_dir(directory)
        .env("RUNMAT_CONFIG", config)
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            directory.join("execution-state"),
        )
        .env("NO_GUI", "1");
    command
}

#[test]
fn unmanifested_compatible_module_runs_in_the_same_binary_isolated_host() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_bridge.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

static void record_clear(void) {
    mxArray *value = mxCreateDoubleScalar(77.0);
    if (mexPutVariable("caller", "cleared_value", value) != 0) {
        mexErrMsgTxt("clear callback failed");
    }
    mxDestroyArray(value);
}

static unsigned int invocation_count = 0;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || nlhs != 1 || !mxIsUint64(prhs[0])) {
        mexErrMsgTxt("expected one uint64 input and output");
    }
    mxArray *arguments[2];
    arguments[0] = mxDuplicateArray(prhs[0]);
    arguments[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxGetUint64s(arguments[1])[0] = 1;
    mxArray *sum = NULL;
    if (mexCallMATLAB(1, &sum, 2, arguments, "plus") != 0) {
        mexErrMsgTxt("callback failed");
    }
    if (mexPutVariable("caller", "from_mex", sum) != 0) {
        mexErrMsgTxt("workspace write failed");
    }
    mxDestroyArray(sum);
    plhs[0] = mexGetVariable("caller", "from_mex");
    if (plhs[0] == NULL) {
        mexErrMsgTxt("workspace read failed");
    }
    invocation_count++;
    if (invocation_count == 1) {
        mexLock();
    } else {
        mexUnlock();
    }
    mexAtExit(record_clear);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "x = isolated_bridge(uint64(1:20000)); disp(x(1)); disp(x(20000)); clear mex; z = isolated_bridge(uint64(5)); disp(z); clear mex; eval('disp(cleared_value)');",
    )
    .unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let values = stdout.split_whitespace().collect::<Vec<_>>();
    assert_eq!(values, ["2", "20001", "6", "77"], "stdout={stdout}");
}

#[test]
fn isolated_cpp_worker_can_finish_an_async_engine_call_after_gateway_return() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_async_worker.cpp");
    fs::write(
        &source,
        r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <chrono>
#include <future>
#include <thread>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        if (!worker_.valid()) {
            auto engine = getEngine();
            auto input = inputs[0];
            mexLock();
            worker_ = std::async(std::launch::async,
                [engine, input]() mutable {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    return engine
                        ->fevalAsync<matlab::data::Array>(u"reshape", input, 2,
                                                          1)
                        .get();
                });
            return;
        }
        outputs[0] = worker_.get();
        mexUnlock();
    }

private:
    std::future<matlab::data::Array> worker_;
};
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "x = [5; 10]; isolated_async_worker(x); y = isolated_async_worker(); disp(y);",
    )
    .unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.split_whitespace().collect::<Vec<_>>(), ["5", "10"]);
}

#[test]
fn session_shutdown_runs_isolated_exit_hooks() {
    let directory = tempfile::tempdir().unwrap();
    let marker_path = directory.path().join("shutdown-complete");
    let marker = c_string_path(&marker_path);
    let source = directory.path().join("isolated_shutdown.c");
    fs::write(
        &source,
        format!(
            r#"
#include "mex.h"
#include <stdio.h>

static void record_shutdown(void) {{
    FILE *marker = fopen("{marker}", "wb");
    if (marker != NULL) {{
        fputs("complete", marker);
        fclose(marker);
    }}
}}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {{
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexAtExit(record_shutdown);
    mexLock();
}}
"#
        ),
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(directory.path().join("main.m"), "isolated_shutdown();").unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(marker_path.is_file());
}

#[test]
fn crashing_compatible_module_cannot_terminate_the_driver() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_crash.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
#include <stdlib.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    abort();
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_crash();",
    )
    .unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RunMat:MEX:HostCrashed"), "stderr={stderr}");
}

#[test]
fn incompatible_module_abi_is_rejected_inside_the_isolated_host() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_incompatible.c");
    fs::write(
        &source,
        r#"
#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif
TEST_EXPORT void mexFunction(int nlhs, void **plhs, int nrhs, const void **prhs) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
    )
    .unwrap();
    let plan = MexBuild::new(&source, directory.path()).plan().unwrap();
    let arguments = plan
        .arguments
        .iter()
        .filter(|argument| {
            std::path::Path::new(argument)
                .file_name()
                .is_none_or(|name| name != "runmat_mex_shim.c")
        })
        .collect::<Vec<_>>();
    let compiler = Command::new(&plan.compiler)
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        compiler.status.success(),
        "compiler stdout={}\ncompiler stderr={}",
        String::from_utf8_lossy(&compiler.stdout),
        String::from_utf8_lossy(&compiler.stderr)
    );
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_incompatible();",
    )
    .unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RunMat:MEX:Load"), "stderr={stderr}");
    assert!(
        stderr.contains("does not expose RunMat's compatibility interface"),
        "stderr={stderr}"
    );
}

#[cfg(unix)]
#[test]
fn missing_native_dependency_is_reported_by_name() {
    let directory = tempfile::tempdir().unwrap();
    let helper_source = directory.path().join("isolated_helper.c");
    let helper_library = directory.path().join(if cfg!(target_os = "macos") {
        "libisolated_helper.dylib"
    } else {
        "libisolated_helper.so"
    });
    fs::write(
        &helper_source,
        "double isolated_helper_value(void) { return 19.0; }",
    )
    .unwrap();
    let source = directory.path().join("isolated_dependency.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
extern double isolated_helper_value(void);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nrhs; (void)prhs;
    if (nlhs > 0) plhs[0] = mxCreateDoubleScalar(isolated_helper_value());
}
"#,
    )
    .unwrap();
    let compiler_path = MexBuild::new(&source, directory.path())
        .plan()
        .unwrap()
        .compiler;
    let mut helper_command = Command::new(compiler_path);
    helper_command
        .arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        })
        .args(["-fPIC", "-o"])
        .arg(&helper_library)
        .arg(&helper_source);
    let helper_build = helper_command.output().unwrap();
    assert!(
        helper_build.status.success(),
        "compiler stdout={}\ncompiler stderr={}",
        String::from_utf8_lossy(&helper_build.stdout),
        String::from_utf8_lossy(&helper_build.stderr)
    );
    let artifact = MexBuild::new(&source, directory.path())
        .linker_argument(helper_library.display().to_string())
        .compile()
        .unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::remove_file(&helper_library).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_dependency();",
    )
    .unwrap();

    let output = run_script(directory.path(), "main.m", "");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RunMat:MEX:Load"), "stderr={stderr}");
    assert!(stderr.contains("dependency:"), "stderr={stderr}");
    assert!(stderr.contains("libisolated_helper"), "stderr={stderr}");
}

#[test]
fn unmanifested_policy_can_deny_compatible_modules_before_host_startup() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_denied.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_denied();",
    )
    .unwrap();

    let output = run_script(
        directory.path(),
        "main.m",
        "[runtime.foreign.mex]\nunmanifested = \"deny\"",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RunMat:MEX:UnmanifestedBinaryDenied"),
        "stderr={stderr}"
    );
}

#[test]
fn timed_out_compatible_module_is_terminated() {
    let _spinning_host = serialize_spinning_host_test();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_timeout.c");
    fs::write(
        &source,
        r#"
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    volatile unsigned long spin = 0;
    for (;;) spin++;
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_timeout();",
    )
    .unwrap();

    let output = run_script(
        directory.path(),
        "main.m",
        "[runtime.foreign.mex]\ntimeout_ms = 100",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RunMat:MEX:Timeout"), "stderr={stderr}");
}

#[test]
fn timed_out_exit_hook_is_terminated_during_clear() {
    let _spinning_host = serialize_spinning_host_test();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("isolated_exit_timeout.c");
    fs::write(
        &source,
        r#"
#include "mex.h"

static void hang_during_clear(void) {
    volatile unsigned long spin = 0;
    for (;;) spin++;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    mexAtExit(hang_during_clear);
}
"#,
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "isolated_exit_timeout(); clear mex;",
    )
    .unwrap();

    let output = run_script(
        directory.path(),
        "main.m",
        "[runtime.foreign.mex]\ntimeout_ms = 100",
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RunMat:MEX:Timeout"), "stderr={stderr}");
}

#[cfg(unix)]
#[test]
fn cancelling_a_compatible_module_terminates_its_host() {
    let _spinning_host = serialize_spinning_host_test();
    let directory = tempfile::tempdir().unwrap();
    let readiness = directory.path().join("invocation-ready");
    let readiness_c_string = c_string_path(&readiness);
    let source = directory.path().join("isolated_cancel.c");
    fs::write(
        &source,
        format!(
            r#"
#include "mex.h"
#include <stdio.h>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {{
    (void)nlhs; (void)plhs; (void)nrhs; (void)prhs;
    FILE *readiness = fopen("{readiness_c_string}", "wb");
    if (readiness == NULL) mexErrMsgTxt("could not record invocation readiness");
    fclose(readiness);
    volatile unsigned long spin = 0;
    for (;;) spin++;
}}
"#
        ),
    )
    .unwrap();
    let artifact = MexBuild::new(&source, directory.path()).compile().unwrap();
    fs::remove_file(&artifact.manifest).unwrap();
    fs::write(
        directory.path().join("main.m"),
        "result = isolated_cancel();",
    )
    .unwrap();

    let mut child = configured_command(directory.path(), "main.m", "")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !readiness.is_file() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if !readiness.is_file() {
        let _ = child.kill();
        let output = child.wait_with_output().unwrap();
        panic!(
            "isolated invocation did not start: status={} stdout={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // SAFETY: `child.id()` is the live process created immediately above.
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("RunMat:MEX:Cancelled"),
        "status={} stderr={stderr}",
        output.status
    );
}
