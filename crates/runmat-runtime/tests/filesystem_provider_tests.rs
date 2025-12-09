use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_filesystem::{self as vfs, SandboxFsProvider};
#[cfg(not(target_arch = "wasm32"))]
use runmat_filesystem::{RemoteFsConfig, RemoteFsProvider};
use runmat_runtime::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_runtime::call_builtin;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn sandbox_provider_supports_repl_and_tabular_builtins() {
    let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
    let temp = tempdir().expect("temp dir");
    let sandbox = Arc::new(
        SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox provider must init"),
    );
    let _guard = vfs::replace_provider(sandbox);

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
    let server = remote_test_support::RemoteTestServer::spawn();
    let remote = Arc::new(
        RemoteFsProvider::new(RemoteFsConfig {
            base_url: server.base_url(),
            auth_token: None,
            chunk_bytes: 1_024,
            parallel_requests: 2,
            timeout: Duration::from_secs(30),
        })
        .expect("remote provider init"),
    );
    let _guard = vfs::replace_provider(remote);

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
    use std::net::TcpListener as StdTcpListener;
    use std::path::PathBuf;
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
                .route("/fs/read", get(read_handler))
                .route("/fs/write", put(write_handler))
                .route("/fs/mkdir", post(mkdir_handler))
                .route("/fs/file", delete(delete_file_handler))
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
            Self {
                root: Arc::new(root),
            }
        }

        fn resolve(&self, remote: &str) -> PathBuf {
            let trimmed = remote.trim_start_matches('/');
            self.root.join(trimmed)
        }
    }

    #[derive(Deserialize)]
    struct PathParams {
        path: String,
        offset: Option<u64>,
        length: Option<usize>,
        truncate: Option<String>,
    }

    #[derive(Deserialize)]
    struct MkdirRequest {
        path: String,
        recursive: bool,
    }

    async fn metadata_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> Json<serde_json::Value> {
        let stats = std::fs::metadata(harness.resolve(&params.path)).unwrap();
        Json(serde_json::json!({
            "fileType": if stats.is_dir() { "dir" } else { "file" },
            "len": stats.len(),
            "modified": stats.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()),
            "readonly": stats.permissions().readonly()
        }))
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

    async fn delete_file_handler(
        State(harness): State<Harness>,
        Query(params): Query<PathParams>,
    ) -> StatusCode {
        let target = harness.resolve(&params.path);
        let _ = std::fs::remove_file(target);
        StatusCode::NO_CONTENT
    }
}
