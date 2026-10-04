use super::*;
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
        Value::Num(fid as f64),
        Value::from("%f %s"),
        vec![Value::Num(1.0)],
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
    assert!(is_shift_jis_encoding("ms932"));
    assert!(is_shift_jis_encoding("x-sjis"));
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
        Value::Num(fid as f64),
        Value::from("%f %s"),
        vec![Value::Num(1.0)],
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
        Value::Num(0.0),
        Value::from("%f"),
        Vec::new(),
    ))
    .expect_err("standard stream rejected");
    assert!(err.message().contains("standard input/output"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn excessive_output_request_rejects_before_text_or_file_scan() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let err = block_on(textscan_builtin(
        Value::from("not a file and not parsed"),
        Value::from("%f"),
        Vec::new(),
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
        Value::Num(fid as f64),
        Value::from("%f %s"),
        vec![Value::from("HeaderLines"), Value::Num(1.0)],
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
