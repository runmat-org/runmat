#![cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

static ISOLATED_NATIVE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fixture_compiler() -> &'static str {
    if cfg!(target_os = "windows") {
        "gcc"
    } else {
        "clang"
    }
}

fn compile_fixture(directory: &Path) -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-runtime/tests/fixtures/native_ffi/interface.c");
    let console = directory.join("console.c");
    fs::write(
        &console,
        r#"#include <stdio.h>
void fixture_console(void) {
    puts("native-stdout");
    fflush(stdout);
    fputs("native-stderr\n", stderr);
    fflush(stderr);
}
"#,
    )
    .expect("write console fixture");
    fs::copy(source.with_extension("h"), directory.join("interface.h"))
        .expect("copy native fixture header");
    fs::write(
        directory.join("console.h"),
        "#include \"interface.h\"\nvoid fixture_console(void);\n",
    )
    .expect("write console fixture header");
    let output = directory.join(if cfg!(target_os = "macos") {
        "libisolated_fixture.dylib"
    } else if cfg!(target_os = "windows") {
        "isolated_fixture.dll"
    } else {
        "libisolated_fixture.so"
    });
    let mut command = Command::new(fixture_compiler());
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else if cfg!(target_os = "windows") {
        command.arg("-shared");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    let compiled = command
        .arg(source)
        .arg(console)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("invoke available native fixture compiler");
    assert!(
        compiled.status.success(),
        "native fixture compilation failed: stdout={}\nstderr={}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    output
}

fn run_script(directory: &Path, source: &str, native_config: &str) -> std::process::Output {
    fs::write(directory.join("main.m"), source).unwrap();
    let config = directory.join("runmat.toml");
    fs::write(
        &config,
        format!(
            r#"
[runtime.accelerate]
enabled = false
provider = "inprocess"

{native_config}
"#
        ),
    )
    .unwrap();
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .arg(directory.join("main.m"))
        .current_dir(directory)
        .env("RUNMAT_CONFIG", config)
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            directory.join("execution-state"),
        )
        .env("NO_GUI", "1")
        .output()
        .unwrap()
}

fn quoted(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

#[test]
fn default_process_host_preserves_calls_callbacks_and_pointer_identity() {
    let _guard = ISOLATED_NATIVE_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if Command::new(fixture_compiler())
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!(
            "skipping isolated native FFI test because {} is unavailable",
            fixture_compiler()
        );
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let library = compile_fixture(directory.path());
    let header = directory.path().join("console.h");
    let source = format!(
        "loadlibrary('{}', '{}', 'alias', 'isolated_fixture', 'addheader', '{}');\n\
         calllib('isolated_fixture', 'fixture_console');\n\
         total = calllib('isolated_fixture', 'fixture_add', int32(19), int32(23));\n\
         pointer = libpointer('int32Ptr', int32(40));\n\
         pointer = calllib('isolated_fixture', 'fixture_increment', pointer);\n\
         opaque = calllib('isolated_fixture', 'fixture_borrowed_value', int32(1));\n\
         setdatatype(opaque, 'int32Ptr', 1, 1);\n\
         callback_total = calllib('isolated_fixture', 'fixture_apply', int32(10), @(value) value + int32(7));\n\
         disp(total); disp(pointer.Value); disp(opaque.Value); disp(callback_total);",
        quoted(&library),
        quoted(&header),
        quoted(&directory.path().join("interface.h"))
    );
    let output = run_script(directory.path(), &source, "");
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .split_whitespace()
            .collect::<Vec<_>>(),
        ["native-stdout", "42", "41", "17", "35"]
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("native-stderr"));
}

#[test]
fn crash_and_timeout_terminate_only_the_native_host() {
    let _guard = ISOLATED_NATIVE_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if Command::new(fixture_compiler())
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!(
            "skipping isolated native FFI test because {} is unavailable",
            fixture_compiler()
        );
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let library = compile_fixture(directory.path());
    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-runtime/tests/fixtures/native_ffi/interface.h");
    let prefix = format!(
        "loadlibrary('{}', '{}', 'alias', 'isolated_fixture');\n",
        quoted(&library),
        quoted(&header)
    );
    let crashed = run_script(
        directory.path(),
        &format!("{prefix}calllib('isolated_fixture', 'fixture_abort');"),
        "",
    );
    assert!(!crashed.status.success());
    assert!(
        String::from_utf8_lossy(&crashed.stderr).contains("HostCrashed"),
        "stderr={}",
        String::from_utf8_lossy(&crashed.stderr)
    );

    let timed_out = run_script(
        directory.path(),
        &format!("{prefix}calllib('isolated_fixture', 'fixture_hang');"),
        "[runtime.foreign.native]\ntimeout_ms = 100",
    );
    assert!(!timed_out.status.success());
    assert!(
        String::from_utf8_lossy(&timed_out.stderr).contains("Timeout"),
        "stderr={}",
        String::from_utf8_lossy(&timed_out.stderr)
    );
}
