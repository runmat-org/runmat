#![cfg(not(target_arch = "wasm32"))]

use pollster::block_on;
use runmat_builtins::{CellArray, NumericDType, Tensor, Value};
use runmat_filesystem::{self as vfs, SandboxFsProvider};
#[cfg(not(target_arch = "wasm32"))]
use runmat_filesystem::{RemoteFsConfig, RemoteFsProvider};
use runmat_runtime::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_runtime::call_builtin;
use runmat_runtime::call_builtin_async_with_outputs;
use std::convert::TryFrom;
use std::io::{Cursor, Write};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use tempfile::tempdir;
use zip::write::SimpleFileOptions;

fn assert_status_one(value: Value, label: &str) {
    assert_eq!(value, Value::Num(1.0), "{label} should return success");
}

fn assert_path_value(value: Value, expected: &str, label: &str) {
    let actual = String::try_from(&value).unwrap_or_else(|_| {
        panic!("{label} should return a path string, got {value:?}");
    });
    assert_eq!(actual.replace('\\', "/"), expected, "{label}");
}

fn assert_string_contains(value: Value, expected: &str, label: &str) {
    let actual = String::try_from(&value).unwrap_or_else(|_| {
        panic!("{label} should return a string, got {value:?}");
    });
    assert!(
        actual.replace('\\', "/").contains(expected),
        "{label} should contain {expected:?}, got {actual:?}"
    );
}

fn exercise_repl_filesystem_builtins_through_provider() {
    assert_path_value(
        call_builtin("pwd", &[]).expect("pwd succeeds"),
        "/",
        "initial pwd",
    );

    assert_status_one(
        call_builtin("mkdir", &[Value::from("workspace")]).expect("mkdir workspace succeeds"),
        "mkdir workspace",
    );
    assert_status_one(
        call_builtin("mkdir", &[Value::from("workspace/nested")]).expect("mkdir nested succeeds"),
        "mkdir nested",
    );

    assert_path_value(
        call_builtin("cd", &[Value::from("workspace")]).expect("cd workspace succeeds"),
        "/",
        "cd should return previous directory",
    );
    assert_path_value(
        call_builtin("pwd", &[]).expect("pwd after cd succeeds"),
        "/workspace",
        "pwd after cd",
    );
    block_on(vfs::write_async("cwd-relative.txt", b"cwd relative"))
        .expect("write relative file through provider cwd");
    let relative = block_on(vfs::read_to_string_async("/workspace/cwd-relative.txt"))
        .expect("read relative file through provider root path");
    assert_eq!(relative, "cwd relative");
    assert_path_value(
        call_builtin("cd", &[Value::from("..")]).expect("cd parent succeeds"),
        "/workspace",
        "cd parent should return previous directory",
    );
    assert_path_value(
        call_builtin("pwd", &[]).expect("pwd after cd parent succeeds"),
        "/",
        "pwd after cd parent",
    );

    assert_status_one(
        call_builtin("mkdir", &[Value::from("/rooted")]).expect("mkdir rooted succeeds"),
        "mkdir rooted",
    );
    assert_path_value(
        call_builtin("cd", &[Value::from("/rooted")]).expect("cd rooted succeeds"),
        "/",
        "cd rooted should return previous directory",
    );
    assert_path_value(
        call_builtin("pwd", &[]).expect("pwd after rooted cd succeeds"),
        "/rooted",
        "pwd after rooted cd",
    );
    block_on(vfs::write_async("/rooted/file.txt", b"rooted path"))
        .expect("write rooted file through provider");
    let rooted = block_on(vfs::read_to_string_async("/rooted/file.txt"))
        .expect("read rooted file through provider");
    assert_eq!(rooted, "rooted path");
    call_builtin("dir", &[Value::from("/rooted")]).expect("dir rooted succeeds");
    assert_string_contains(
        call_builtin("genpath", &[Value::from("/rooted")]).expect("genpath rooted succeeds"),
        "/rooted",
        "genpath rooted",
    );
    call_builtin("addpath", &[Value::from("/rooted")]).expect("addpath rooted succeeds");
    call_builtin("rmpath", &[Value::from("/rooted")]).expect("rmpath rooted succeeds");
    assert_path_value(
        call_builtin("cd", &[Value::from("/")]).expect("cd root succeeds"),
        "/rooted",
        "cd root should return previous directory",
    );

    block_on(vfs::write_async("workspace/source.txt", b"provider text"))
        .expect("write source file through provider");
    assert_status_one(
        call_builtin(
            "copyfile",
            &[
                Value::from("workspace/source.txt"),
                Value::from("workspace/copied.txt"),
            ],
        )
        .expect("copyfile succeeds"),
        "copyfile",
    );
    let copied = block_on(vfs::read_to_string_async("workspace/copied.txt"))
        .expect("read copied file through provider");
    assert_eq!(copied, "provider text");

    assert_status_one(
        call_builtin(
            "movefile",
            &[
                Value::from("workspace/copied.txt"),
                Value::from("workspace/moved.txt"),
            ],
        )
        .expect("movefile succeeds"),
        "movefile",
    );
    assert!(
        block_on(vfs::metadata_async("workspace/copied.txt")).is_err(),
        "movefile should remove the original path"
    );
    let moved = block_on(vfs::read_to_string_async("workspace/moved.txt"))
        .expect("read moved file through provider");
    assert_eq!(moved, "provider text");

    call_builtin("dir", &[Value::from("workspace")]).expect("dir succeeds");
    call_builtin("ls", &[Value::from("workspace")]).expect("ls succeeds");

    call_builtin("delete", &[Value::from("workspace/moved.txt")]).expect("delete succeeds");
    assert!(
        block_on(vfs::metadata_async("workspace/moved.txt")).is_err(),
        "delete should remove the file"
    );
    assert_status_one(
        call_builtin("rmdir", &[Value::from("workspace"), Value::from("s")])
            .expect("rmdir recursive succeeds"),
        "rmdir",
    );
    assert!(
        block_on(vfs::metadata_async("workspace")).is_err(),
        "rmdir should remove the directory tree"
    );
}

fn zip_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let cursor = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in entries {
        zip.start_file(name, options).expect("start zip entry");
        zip.write_all(bytes).expect("write zip entry");
    }
    zip.finish().expect("finish zip").into_inner()
}

fn wav_bytes(sample_rate: u32, channels: u16, bits: u16, frames: u32) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_size = frames * block_align as u32;
    let riff_size = 36 + data_size;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.resize(bytes.len() + data_size as usize, 0);
    bytes
}

fn assert_struct_field(value: &Value, name: &str, expected: &Value) {
    let Value::Struct(fields) = value else {
        panic!("expected struct output, got {value:?}");
    };
    assert_eq!(
        fields.fields.get(name),
        Some(expected),
        "field {name} should match"
    );
}

fn exercise_recent_import_export_builtins_through_provider() {
    call_builtin("mkdir", &[Value::from("fixtures")]).expect("mkdir fixtures succeeds");
    call_builtin("mkdir", &[Value::from("exports")]).expect("mkdir exports succeeds");

    block_on(vfs::write_async("fixtures/numeric.csv", b"1,2\n3,4\n"))
        .expect("write importdata fixture through provider");
    let imported = call_builtin(
        "importdata",
        &[Value::from("fixtures/numeric.csv"), Value::from(",")],
    )
    .expect("importdata succeeds");
    assert_eq!(
        imported,
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap())
    );

    let cell = CellArray::new(
        vec![
            Value::Num(1.0),
            Value::from("alpha"),
            Value::Bool(true),
            Value::Num(4.0),
        ],
        2,
        2,
    )
    .expect("cell array");
    call_builtin(
        "writecell",
        &[
            Value::Cell(cell),
            Value::from("exports/cells.csv"),
            Value::from("Delimiter"),
            Value::from(","),
        ],
    )
    .expect("writecell succeeds");
    let written = block_on(vfs::read_to_string_async("exports/cells.csv"))
        .expect("read writecell output through provider");
    assert_eq!(written, "1,\"alpha\"\n1,4\n");

    block_on(vfs::write_async(
        "fixtures/textscan.txt",
        b"10 ten\n20 twenty\n",
    ))
    .expect("write textscan fixture through provider");
    let fid = block_on(call_builtin_async_with_outputs(
        "fopen",
        &[Value::from("fixtures/textscan.txt"), Value::from("r")],
        1,
    ))
    .expect("fopen succeeds");
    let Value::OutputList(open_values) = fid else {
        panic!("expected fopen output list");
    };
    let Value::Num(fid) = open_values[0] else {
        panic!("expected numeric fid");
    };
    let scanned = call_builtin("textscan", &[Value::Num(fid), Value::from("%f %s")])
        .expect("textscan succeeds");
    let Value::Cell(scanned_cells) = scanned else {
        panic!("expected textscan cell output");
    };
    assert_eq!(
        scanned_cells.get(0, 0).expect("numeric column"),
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap())
    );
    let Value::Cell(words) = scanned_cells.get(0, 1).expect("text column") else {
        panic!("expected string column cell");
    };
    assert_eq!(words.get(0, 0).unwrap(), Value::from("ten"));
    assert_eq!(words.get(1, 0).unwrap(), Value::from("twenty"));
    call_builtin("fclose", &[Value::Num(fid)]).expect("fclose succeeds");

    block_on(vfs::write_async(
        "fixtures/sound.wav",
        wav_bytes(44_100, 2, 16, 4),
    ))
    .expect("write wav fixture through provider");
    let info = call_builtin("audioinfo", &[Value::from("fixtures/sound.wav")])
        .expect("audioinfo succeeds");
    assert_struct_field(&info, "Format", &Value::from("WAV"));
    assert_struct_field(&info, "SampleRate", &Value::Num(44_100.0));
    assert_struct_field(&info, "NumChannels", &Value::Num(2.0));

    block_on(vfs::write_async(
        "fixtures/archive.zip",
        zip_bytes(&[("nested/data.txt", b"zipped")]),
    ))
    .expect("write zip fixture through provider");
    call_builtin(
        "unzip",
        &[
            Value::from("fixtures/archive.zip"),
            Value::from("exports/unzipped"),
        ],
    )
    .expect("unzip succeeds");
    let extracted = block_on(vfs::read_to_string_async(
        "exports/unzipped/nested/data.txt",
    ))
    .expect("read unzip output through provider");
    assert_eq!(extracted, "zipped");
}

#[test]
fn sandbox_provider_supports_repl_and_tabular_builtins() {
    let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
    let _provider_lock = vfs::provider_override_lock();
    let temp = tempdir().expect("temp dir");
    let sandbox = Arc::new(
        SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox provider must init"),
    );
    let _guard = vfs::replace_provider(sandbox);

    exercise_repl_filesystem_builtins_through_provider();
    exercise_recent_import_export_builtins_through_provider();

    call_builtin("mkdir", &[Value::from("reports")]).expect("mkdir succeeds");

    let matrix = Tensor {
        data: vec![1.0, 2.0, 3.5, 4.0],
        shape: vec![2, 2],
        rows: 2,
        cols: 2,
        dtype: NumericDType::F64,
    };
    let filename = Value::from("reports/data.csv".to_string());
    call_builtin(
        "dlmwrite",
        &[filename.clone(), Value::Tensor(matrix.clone())],
    )
    .expect("dlmwrite succeeds");

    let host_path = temp.path().join("reports").join("data.csv");
    assert!(
        host_path.exists(),
        "backing filesystem should observe sandbox writes"
    );

    let read_back = call_builtin("dlmread", &[filename]).expect("dlmread succeeds");
    assert_eq!(read_back, Value::Tensor(matrix));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn remote_provider_supports_repl_and_tabular_builtins() {
    let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
    let _provider_lock = vfs::provider_override_lock();
    let server = remote_test_support::RemoteTestServer::spawn();
    let remote = Arc::new(
        RemoteFsProvider::new(RemoteFsConfig {
            base_url: server.base_url(),
            auth_token: None,
            chunk_bytes: 1_024,
            parallel_requests: 2,
            direct_read_threshold_bytes: 64 * 1024 * 1024,
            timeout: Duration::from_secs(30),
            ..RemoteFsConfig::default()
        })
        .expect("remote provider init"),
    );
    let _guard = vfs::replace_provider(remote);

    exercise_repl_filesystem_builtins_through_provider();

    call_builtin("mkdir", &[Value::from("remote")]).expect("mkdir succeeds");
    let matrix = Tensor {
        data: vec![10.0, 20.0, 30.0, 40.0],
        shape: vec![2, 2],
        rows: 2,
        cols: 2,
        dtype: NumericDType::F64,
    };
    let filename = Value::from("remote/data.csv".to_string());
    call_builtin(
        "dlmwrite",
        &[filename.clone(), Value::Tensor(matrix.clone())],
    )
    .expect("dlmwrite succeeds");

    let roundtrip = call_builtin("dlmread", &[filename]).expect("dlmread succeeds");
    assert_eq!(roundtrip, Value::Tensor(matrix));
    server.shutdown();
}

#[cfg(not(target_arch = "wasm32"))]
mod remote_test_support {
    use super::*;
    use axum::extract::{Query, State};
    use axum::http::StatusCode;
    use axum::routing::{delete, get, post, put};
    use axum::{Json, Router};
    use serde::Deserialize;
    use std::ffi::OsString;
    use std::net::TcpListener as StdTcpListener;
    use std::path::{Component, Path, PathBuf};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::net::TcpListener as TokioTcpListener;
    use tokio::runtime::Runtime;

    pub(super) struct RemoteTestServer {
        addr: String,
        _temp: TempDir,
        runtime: Runtime,
    }

    impl RemoteTestServer {
        pub fn spawn() -> Self {
            let temp = tempdir().expect("tempdir");
            let harness = Harness::new(temp.path().to_path_buf());
            let router = Router::new()
                .route("/fs/metadata", get(metadata_handler))
                .route("/fs/canonicalize", get(canonicalize_handler))
                .route("/fs/dir", get(read_dir_handler).delete(delete_dir_handler))
                .route("/fs/read", get(read_handler))
                .route("/fs/write", put(write_handler))
                .route("/fs/mkdir", post(mkdir_handler))
                .route("/fs/file", delete(delete_file_handler))
                .route("/fs/rename", post(rename_handler))
                .with_state(harness);
            let std_listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
            std_listener.set_nonblocking(true).expect("nonblocking");
            let addr = std_listener.local_addr().unwrap();
            let runtime = Runtime::new().expect("runtime");
            runtime.spawn(async move {
                let listener = TokioTcpListener::from_std(std_listener).unwrap();
                axum::serve(listener, router.into_make_service())
                    .await
                    .unwrap();
            });
            Self {
                addr: format!("http://{}", addr),
                _temp: temp,
                runtime,
            }
        }

        pub fn base_url(&self) -> String {
            self.addr.clone()
        }

        pub fn shutdown(self) {
            self.runtime.shutdown_background();
        }
    }

    #[derive(Clone)]
    struct Harness {
        root: Arc<PathBuf>,
    }

    impl Harness {
        fn new(root: PathBuf) -> Self {
            let root = std::fs::canonicalize(&root).unwrap_or(root);
            Self {
                root: Arc::new(root),
            }
        }

        fn resolve(&self, remote: &str) -> PathBuf {
            let mut segments: Vec<OsString> = Vec::new();
            for component in Path::new(remote).components() {
                match component {
                    Component::Prefix(_) | Component::RootDir | Component::CurDir => {}
                    Component::ParentDir => {
                        segments.pop();
                    }
                    Component::Normal(segment) => segments.push(segment.to_os_string()),
                }
            }
            let mut target = (*self.root).clone();
            for segment in segments {
                target.push(segment);
            }
            target
        }

        fn virtualize(&self, real: &Path) -> String {
            let relative = real.strip_prefix(&*self.root).unwrap_or(Path::new(""));
            if relative.as_os_str().is_empty() {
                "/".to_string()
            } else {
                format!("/{}", relative.to_string_lossy().replace('\\', "/"))
            }
        }
    }

    #[derive(Deserialize)]
    struct PathParams {
        path: String,
        offset: Option<u64>,
        length: Option<usize>,
        truncate: Option<String>,
        recursive: Option<String>,
    }

    #[derive(Deserialize)]
    struct MkdirRequest {
        path: String,
        recursive: bool,
    }

    #[derive(Deserialize)]
    struct RenameRequest {
        from: String,
        to: String,
    }

    async fn metadata_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let stats =
            std::fs::metadata(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        Ok(Json(serde_json::json!({
            "fileType": if stats.is_dir() { "dir" } else { "file" },
            "len": stats.len(),
            "modified": stats.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()),
            "readonly": stats.permissions().readonly()
        })))
    }

    async fn read_dir_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
        let read_dir =
            std::fs::read_dir(harness.resolve(&params.path)).map_err(|_| StatusCode::NOT_FOUND)?;
        let mut entries = Vec::new();
        for entry in read_dir {
            let entry = entry.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            let path = entry.path();
            let stats = entry
                .metadata()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            entries.push(serde_json::json!({
                "path": harness.virtualize(&path),
                "fileName": entry.file_name().to_string_lossy(),
                "fileType": if stats.is_dir() { "dir" } else { "file" }
            }));
        }
        Ok(Json(entries))
    }

    async fn canonicalize_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        let path = std::fs::canonicalize(harness.resolve(&params.path))
            .map_err(|_| StatusCode::NOT_FOUND)?;
        Ok(Json(serde_json::json!({
            "path": harness.virtualize(&path)
        })))
    }

    async fn read_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> (axum::http::HeaderMap, Vec<u8>) {
        let mut data = std::fs::read(harness.resolve(&params.path)).unwrap();
        let offset = params.offset.unwrap_or(0) as usize;
        let length = params.length.unwrap_or(data.len().saturating_sub(offset));
        let end = std::cmp::min(offset + length, data.len());
        data = data[offset..end].to_vec();
        (axum::http::HeaderMap::new(), data)
    }

    async fn write_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
        body: axum::body::Bytes,
    ) -> StatusCode {
        let target = harness.resolve(&params.path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let truncate = params.truncate.as_deref() == Some("true");
        let mut data = if truncate || !target.exists() {
            Vec::new()
        } else {
            std::fs::read(&target).unwrap()
        };
        let offset = params.offset.unwrap_or(0) as usize;
        let required = offset + body.len();
        if required > data.len() {
            data.resize(required, 0);
        }
        data[offset..offset + body.len()].copy_from_slice(&body);
        std::fs::write(target, data).unwrap();
        StatusCode::NO_CONTENT
    }

    async fn mkdir_handler(
        State(harness): State<Harness>,
        Json(payload): Json<MkdirRequest>,
    ) -> StatusCode {
        let target = harness.resolve(&payload.path);
        if payload.recursive {
            std::fs::create_dir_all(target).unwrap();
        } else {
            std::fs::create_dir(target).unwrap();
        }
        StatusCode::NO_CONTENT
    }

    async fn delete_dir_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> StatusCode {
        let target = harness.resolve(&params.path);
        let recursive = params.recursive.as_deref() == Some("true");
        let result = if recursive {
            std::fs::remove_dir_all(target)
        } else {
            std::fs::remove_dir(target)
        };
        match result {
            Ok(()) => StatusCode::NO_CONTENT,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    async fn delete_file_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> StatusCode {
        let target = harness.resolve(&params.path);
        let _ = std::fs::remove_file(target);
        StatusCode::NO_CONTENT
    }

    async fn rename_handler(
        State(harness): State<Harness>,
        Json(payload): Json<RenameRequest>,
    ) -> StatusCode {
        let source = harness.resolve(&payload.from);
        let target = harness.resolve(&payload.to);
        if !source.exists() {
            return StatusCode::NOT_FOUND;
        }
        if let Some(parent) = target.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return StatusCode::INTERNAL_SERVER_ERROR;
            }
        }
        match std::fs::rename(source, target) {
            Ok(()) => StatusCode::NO_CONTENT,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
