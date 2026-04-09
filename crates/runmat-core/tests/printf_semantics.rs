#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_core::{ExecutionResult, ExecutionStreamKind, RunMatSession};
use runmat_gc::gc_test_context;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn stdout_stream(result: &ExecutionResult) -> String {
    result
        .streams
        .iter()
        .filter(|entry| entry.stream == ExecutionStreamKind::Stdout)
        .map(|entry| entry.text.as_str())
        .collect::<String>()
}

fn stderr_stream(result: &ExecutionResult) -> String {
    result
        .streams
        .iter()
        .filter(|entry| entry.stream == ExecutionStreamKind::Stderr)
        .map(|entry| entry.text.as_str())
        .collect::<String>()
}

fn unique_temp_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("runmat_{prefix}_{nanos}.txt"))
}

#[test]
fn fprintf_rm138_repro_is_stable_end_to_end() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let script = r#"
        price = 42.5;
        fprintf("Price: $%.2f\n", price);
        x = single(3.14);
        fprintf("Value: %.4f\n", double(x));
    "#;
    let result = block_on(engine.execute(script)).unwrap();
    assert_eq!(stdout_stream(&result), "Price: $42.50\nValue: 3.1400\n");
}

#[test]
fn fprintf_stream_routing_is_correct() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let result = block_on(engine.execute("fprintf('out'); fprintf(2, 'err');")).unwrap();
    assert_eq!(stdout_stream(&result), "out");
    assert_eq!(stderr_stream(&result), "err");
}

#[test]
fn fprintf_grouping_and_i_flag_smoke() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let result = block_on(engine.execute("fprintf('%''d|%Id', 12345, 42);")).unwrap();
    assert_eq!(stdout_stream(&result), "12,345|42");
}

#[test]
fn fprintf_file_roundtrip_and_count_work_end_to_end() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let path = unique_temp_path("fprintf_core_roundtrip");
    let path_text = path.to_string_lossy();
    let script = format!(
        "fid = fopen('{}', 'w'); n = fprintf(fid, 'hello-%d', 7); fclose(fid); fprintf('|%d|', n);",
        path_text
    );
    let result = block_on(engine.execute(&script)).unwrap();
    assert_eq!(stdout_stream(&result), "|7|");
    let bytes = std::fs::read(&path).expect("written file should exist");
    assert_eq!(bytes, b"hello-7");
    let _ = std::fs::remove_file(path);
}

#[test]
fn fprintf_encoding_alias_smoke_utf8_underscore() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let path = unique_temp_path("fprintf_core_utf8_alias");
    let path_text = path.to_string_lossy();
    let script = format!(
        "fid = fopen('{}', 'w', 'native', 'utf_8'); fprintf(fid, '%s', 'é'); fclose(fid);",
        path_text
    );
    block_on(engine.execute(&script)).unwrap();
    let bytes = std::fs::read(&path).expect("utf8 alias output should exist");
    assert_eq!(bytes, "é".as_bytes());
    let _ = std::fs::remove_file(path);
}

#[test]
fn fprintf_format_error_propagates_to_session_boundary() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();
    let result = block_on(engine.execute("fprintf('%q', 1);"));
    match result {
        Ok(exec) => {
            let msg = exec
                .error
                .map(|err| err.message().to_string())
                .unwrap_or_default();
            assert!(msg.contains("unsupported format %q"), "{msg}");
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(msg.contains("unsupported format %q"), "{msg}");
        }
    }
}
