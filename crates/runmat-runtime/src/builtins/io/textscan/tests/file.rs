use super::*;

#[path = "file/cooperation.rs"]
mod cooperation;
use crate::builtins::io::filetext::registry::RegisteredFile;
use futures::executor::block_on;
use runmat_filesystem::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};

struct FileFixture {
    path: PathBuf,
    _owner: runmat_filesystem::ProviderGuard,
    _lock: MutexGuard<'static, ()>,
}

impl std::ops::Deref for FileFixture {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for FileFixture {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

fn file_fixture(name: &str, bytes: &[u8]) -> FileFixture {
    let lock = runmat_filesystem::provider_override_lock();
    let path = PathBuf::from(format!("/textscan/{name}"));
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/textscan",
    ));
    provider
        .write_project_path(&path, bytes)
        .expect("write fixture");
    let owner = runmat_filesystem::replace_provider(provider);
    FileFixture {
        path,
        _owner: owner,
        _lock: lock,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn registered_snapshot_retains_exact_metadata_and_observes_close() {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("snapshot.txt", b"10\n");
    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });
    let snapshot = registry::snapshot_file(fid).expect("registered snapshot");
    assert_eq!(snapshot.info.id, fid);
    assert_eq!(snapshot.info.path.as_deref(), Some(path.as_ref()));
    assert_eq!(snapshot.info.access, registry::FileAccess::READ);
    assert!(Arc::ptr_eq(&snapshot.handle, &handle));
    block_on(registry::close_async(fid))
        .expect("close")
        .expect("closed entry");
    assert!(registry::snapshot_file(fid).is_none());
    assert!(snapshot.handle.lock().expect("retained slot").is_none());
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn checked_out_file_is_busy_not_closed_and_drop_restores_exact_cursor() {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("operation-busy.txt", b"first\nsecond\n");
    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });
    let mut operation = registry::acquire_file(fid).expect("checkout");
    assert_eq!(operation.info.id, fid);
    let mut first = [0; 6];
    std::io::Read::read_exact(operation.file_mut(), &mut first).expect("read");
    assert_eq!(&first, b"first\n");
    assert!(matches!(
        registry::acquire_file(fid),
        Err(registry::FileOperationError::Busy)
    ));
    let reader_error = block_on(crate::builtins::io::filetext::fgetl::evaluate(
        &Value::Num(fid as f64),
        &[],
    ))
    .expect_err("a reentrant reader is busy, not closed");
    assert_eq!(reader_error.identifier(), Some("RunMat:fgetl:IoFailure"));
    assert!(reader_error.message().contains("another operation"));
    let close_error = block_on(registry::close_async(fid)).expect_err("busy close");
    assert_eq!(close_error.kind(), std::io::ErrorKind::WouldBlock);
    assert!(registry::snapshot_file(fid).is_some());
    drop(operation);
    assert!(Arc::ptr_eq(
        &handle,
        &registry::snapshot_file(fid).expect("same entry").handle
    ));
    let mut second = String::new();
    let mut resumed = registry::acquire_file(fid).expect("resume");
    std::io::Read::read_to_string(resumed.file_mut(), &mut second).expect("remaining data");
    assert_eq!(second, "second\n");
    drop(resumed);
    block_on(registry::close_async(fid))
        .expect("close after release")
        .expect("entry");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_repeated_file_read_preserves_next_record_position() {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("file-position.txt", b"10 ten\n20 twenty\n");

    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });

    let out = block_on(textscan_builtin(
        None,
        Value::Num(fid as f64),
        Value::from("%f %s"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect("textscan");
    assert_eq!(numeric_column(&out, 0), vec![10.0]);
    assert_eq!(text_column(&out, 1), vec!["ten".to_string()]);

    let mut remaining = String::new();
    let mut guard = handle.lock().expect("lock");
    let file = guard.as_mut().expect("file");
    std::io::Read::read_to_string(file, &mut remaining).expect("read remaining");
    assert_eq!(remaining, "20 twenty\n");

    let _ = registry::close(fid);
    let _ = runmat_filesystem::remove_file(path);
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn initial_utf8_bom_partial_read_seeks_to_exact_source_byte() {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("bom-partial.txt", b"\xef\xbb\xbf10\n20\n");
    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle: handle.clone(),
    });
    let out = block_on(textscan_builtin(
        None,
        Value::Num(fid as f64),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect("read first record");
    assert_eq!(numeric_column(&out, 0), vec![10.0]);
    let mut remaining = String::new();
    let mut operation = registry::acquire_file(fid).expect("same file");
    std::io::Read::read_to_string(operation.file_mut(), &mut remaining).expect("remainder");
    assert_eq!(remaining, "20\n");
    drop(operation);
    block_on(registry::close_async(fid))
        .expect("close")
        .expect("entry");
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_shift_jis_registered_file_restores_source_byte_position() {
    shift_jis_registered_file_restores_source_byte_position(
        "shift_jis",
        &[
            b'1', b' ', 0x82, 0xA0, b'\n', b'2', b' ', b'n', b'e', b'x', b't', b'\n',
        ],
        "あ",
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_shift_jis_alias_restores_source_byte_position() {
    shift_jis_registered_file_restores_source_byte_position(
        "windows-31j",
        &[
            b'1', b' ', 0x82, 0xA0, b'\n', b'2', b' ', b'n', b'e', b'x', b't', b'\n',
        ],
        "あ",
    );
    for alias in ["ms932", "x-sjis"] {
        shift_jis_registered_file_restores_source_byte_position(
            alias,
            &[
                b'1', b' ', 0x82, 0xA0, b'\n', b'2', b' ', b'n', b'e', b'x', b't', b'\n',
            ],
            "あ",
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_shift_jis_0x80_advances_one_source_byte() {
    shift_jis_registered_file_restores_source_byte_position(
        "ms932",
        &[
            b'1', b' ', 0x80, b'\n', b'2', b' ', b'n', b'e', b'x', b't', b'\n',
        ],
        "\u{80}",
    );
}

fn shift_jis_registered_file_restores_source_byte_position(
    encoding: &str,
    bytes: &[u8],
    expected_text: &str,
) {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture(
        &format!("shift-jis-{}.txt", encoding.replace('-', "_")),
        bytes,
    );

    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: encoding.to_string(),
        handle: handle.clone(),
    });

    let out = block_on(textscan_builtin(
        None,
        Value::Num(fid as f64),
        Value::from("%f %s"),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
    ))
    .expect("textscan");
    assert_eq!(numeric_column(&out, 0), vec![1.0]);
    assert_eq!(text_column(&out, 1), vec![expected_text.to_string()]);

    let mut remaining = Vec::new();
    let mut guard = handle.lock().expect("lock");
    let file = guard.as_mut().expect("file");
    std::io::Read::read_to_end(file, &mut remaining).expect("read remaining");
    assert_eq!(remaining, b"2 next\n");

    let _ = registry::close(fid);
    let _ = runmat_filesystem::remove_file(path);
    registry::reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_rejects_standard_stream_identifiers() {
    let err = block_on(textscan_builtin(
        None,
        Value::Num(0.0),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("standard stream rejected");
    assert!(err.message().contains("standard input/output"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn excessive_output_request_rejects_before_text_or_file_scan() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err = block_on(textscan_builtin(
        None,
        Value::from("not a file and not parsed"),
        Value::from("%f"),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("fixed output count");
    assert_eq!(err.identifier(), Some("RunMat:textscan:InvalidArgument"));
    assert!(err.message().contains("at most one output"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn textscan_reads_from_registered_file_identifier() {
    let _guard = registry::test_guard();
    registry::reset_for_tests();
    let path = file_fixture("registered-file.txt", b"skip\n10 ten\n20 twenty\n");

    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async(&path)).expect("open file");
    let handle = Arc::new(StdMutex::new(Some(file)));
    let fid = registry::register_file(RegisteredFile {
        path: path.to_path_buf(),
        permission: "r".to_string(),
        access: registry::FileAccess::READ,
        machinefmt: "native".to_string(),
        encoding: "UTF-8".to_string(),
        handle,
    });

    let out = block_on(textscan_builtin(
        None,
        Value::Num(fid as f64),
        Value::from("%f %s"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("HeaderLines"),
            Value::Num(1.0),
        ]),
    ))
    .expect("textscan");
    assert_eq!(numeric_column(&out, 0), vec![10.0, 20.0]);
    assert_eq!(
        text_column(&out, 1),
        vec!["ten".to_string(), "twenty".to_string()]
    );

    let _ = registry::close(fid);
    let _ = runmat_filesystem::remove_file(path);
    registry::reset_for_tests();
}
