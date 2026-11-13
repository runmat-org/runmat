use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::{fopen, registry};
use crate::RuntimeError;
use runmat_accelerate_api::HostTensorView;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::IntValue;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
    futures::executor::block_on(evaluate(fid_value, rest))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

mod controls;

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}

fn fopen_path(path: &Path) -> FopenHandle {
    let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
    let open = eval.as_open().expect("open outputs");
    assert!(open.fid >= 3.0);
    FopenHandle {
        fid: open.fid as i32,
    }
}

struct FopenHandle {
    fid: i32,
}

impl Drop for FopenHandle {
    fn drop(&mut self) {
        let _ = registry::close(self.fid);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_reads_line_with_newline() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_line");
    test_support::fs::write(&path, "Hello world\nSecond line\n").unwrap();

    let handle = fopen_path(&path);
    let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
    let line = eval.first_output();
    match line {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "Hello world\n");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    let ltout = eval.outputs()[1].clone();
    match ltout {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![10.0]);
            assert_eq!(t.shape(), &[1, 1]);
        }
        other => panic!("expected numeric tensor, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_returns_minus_one_at_eof() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_eof");
    test_support::fs::write(&path, "line\n").unwrap();
    let handle = fopen_path(&path);

    let _ = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
    let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
    assert_eq!(eval.first_output(), Value::Num(-1.0));
    assert_eq!(eval.outputs()[1], Value::Num(-1.0));

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_honours_nchar_limit() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_limit");
    test_support::fs::write(&path, "abcdefghij\nrest\n").unwrap();
    let handle = fopen_path(&path);

    let eval =
        run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(5.0)]).expect("limited read");
    match eval.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "abcde");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    match &eval.outputs()[1] {
        Value::Tensor(t) => {
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected empty numeric tensor, got {other:?}"),
    }

    let next = run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(5.0)]).expect("next read");
    match next.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "fghij");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_errors_for_write_only_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_write_only");
    test_support::fs::write(&path, "payload").unwrap();
    let eval = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("fopen");
    let open = eval.as_open().expect("open outputs");
    assert!(open.fid >= 3.0);
    let err = unwrap_error_message(run_evaluate(&Value::Num(open.fid), &[]).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier is not open for reading",
            FGETS_ERROR_INVALID_IDENTIFIER.message
        )
    );
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_respects_limit_before_crlf_sequence() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_limit_crlf");
    test_support::fs::write(&path, b"ABCDE\r\nnext\n").unwrap();
    let handle = fopen_path(&path);

    let first = run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(3.0)]).expect("first");
    match first.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "ABC");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    match &first.outputs()[1] {
        Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
        other => panic!("expected empty numeric tensor, got {other:?}"),
    }

    let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
    match second.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "DE\r\n");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    match &second.outputs()[1] {
        Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![13.0, 10.0]),
        other => panic!("expected CRLF terminators, got {other:?}"),
    }

    let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
    match third.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "next\n");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_handles_crlf_newlines() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_crlf");
    test_support::fs::write(&path, b"first line\r\nsecond\r\n").unwrap();
    let handle = fopen_path(&path);

    let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
    let outputs = eval.outputs();
    match &outputs[0] {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "first line\r\n");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![13.0, 10.0]);
        }
        other => panic!("expected numeric tensor, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_consumes_lfcr_as_one_newline_sequence() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_lfcr");
    test_support::fs::write(&path, b"first\n\rsecond").unwrap();
    let handle = fopen_path(&path);

    let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
    match first.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "first\n\r");
        }
        other => panic!("expected char array, got {other:?}"),
    }
    let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
    match second.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "second");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_decodes_latin1() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_latin1");
    test_support::fs::write(&path, [0x48u8, 0x6f, 0x6c, 0x61, 0x20, 0xf1, b'\n']).unwrap();
    let eval = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r"),
        Value::from("native"),
        Value::from("latin1"),
    ])
    .expect("fopen");
    let open = eval.as_open().expect("open outputs");
    let fid = open.fid as i32;

    let read = run_evaluate(&Value::Num(fid as f64), &[]).expect("fgets");
    match read.first_output() {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "Hola ñ\n");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = registry::close(fid);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_nchar_zero_returns_empty_char() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_zero");
    test_support::fs::write(&path, "hello\n").unwrap();
    let handle = fopen_path(&path);

    let eval = run_evaluate(
        &Value::Num(handle.fid as f64),
        &[Value::Int(IntValue::I32(0))],
    )
    .expect("fgets");
    match eval.first_output() {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 0);
            assert!(ca.data().is_empty());
        }
        other => panic!("expected empty char array, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_nchar_one_returns_one_character() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_one");
    test_support::fs::write(&path, "hello\n").unwrap();
    let handle = fopen_path(&path);

    let eval = run_evaluate(
        &Value::Num(handle.fid as f64),
        &[Value::Int(IntValue::I32(1))],
    )
    .expect("fgets");
    match eval.first_output() {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 1);
            assert_eq!(ca.data(), runmat_value::encode_utf16_code_units("h"));
        }
        other => panic!("expected one-character row, got {other:?}"),
    }

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgets_gathers_gpu_scalar_arguments() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgets_gpu_args");
    test_support::fs::write(&path, b"abcdef\nextra").unwrap();
    let handle = fopen_path(&path);

    test_support::with_test_provider(|provider| {
        let fid_host = [handle.fid as f64];
        let fid_view = HostTensorView {
            data: &fid_host,
            shape: &[1, 1],
        };
        let fid_gpu = Value::GpuTensor(provider.upload(&fid_view).expect("upload fid"));

        let limit_host = [3.0f64];
        let limit_view = HostTensorView {
            data: &limit_host,
            shape: &[1, 1],
        };
        let limit_gpu = Value::GpuTensor(provider.upload(&limit_view).expect("upload limit"));

        let eval = run_evaluate(&fid_gpu, &[limit_gpu]).expect("fgets");
        match eval.first_output() {
            Value::CharArray(ca) => {
                let text: String =
                    runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
                assert_eq!(text, "abc");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    });

    test_support::fs::remove_file(&path).unwrap();
}
