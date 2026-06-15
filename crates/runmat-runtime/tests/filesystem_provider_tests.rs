#![cfg(not(target_arch = "wasm32"))]

use pollster::block_on;
use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_filesystem::{self as vfs, SandboxFsProvider};
#[cfg(not(target_arch = "wasm32"))]
use runmat_filesystem::{RemoteFsConfig, RemoteFsProvider};
use runmat_runtime::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_runtime::call_builtin;
use std::convert::TryFrom;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use tempfile::tempdir;

fn assert_status_one(value: Value, label: &str) {
    assert_eq!(value, Value::Num(1.0), "{label} should return success");
}

fn assert_path_value(value: Value, expected: &str, label: &str) {
    let actual = String::try_from(&value).unwrap_or_else(|_| {
        panic!("{label} should return a path string, got {value:?}");
    });
    assert_eq!(actual.replace('\\', "/"), expected, "{label}");
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
