use super::*;
#[cfg(not(target_arch = "wasm32"))]
use crate::builtins::io::http::transport::NativeHttpService;
#[cfg(not(target_arch = "wasm32"))]
use crate::context::{RuntimeContext, RuntimeServicePorts};
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tempfile::tempdir;
use zip::write::SimpleFileOptions;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(target_arch = "wasm32"))]
struct FiniteHostBudget;
#[cfg(not(target_arch = "wasm32"))]
impl crate::context::RuntimeParallelService for FiniteHostBudget {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(1024),
            ..Default::default()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn finite_budget_without_host_effect_evidence_rejects_before_filesystem_effects() {
    let root = tempdir().unwrap();
    let archive = root.path().join("missing.zip");
    let destination = root.path().join("must-not-create");
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(FiniteHostBudget)),
        );
    let error = block_on(unzip_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String(archive.to_string_lossy().into_owned().into()),
            Value::String(destination.to_string_lossy().into_owned().into()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert!(!destination.exists());
}

struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl AccelProvider for CountingProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}

fn unique_zip_path(stem: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{id}.zip"))
}

fn write_test_zip(path: &Path, entries: &[(&str, &[u8])]) {
    let file = std::fs::File::create(path).expect("create zip");
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in entries {
        zip.start_file(name, options).expect("start zip entry");
        zip.write_all(bytes).expect("write zip entry");
    }
    zip.finish().expect("finish zip");
}

fn write_test_zip_with_dirs(path: &Path, dirs: &[&str], entries: &[(&str, &[u8])]) {
    let file = std::fs::File::create(path).expect("create zip");
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for name in dirs {
        zip.add_directory(*name, options).expect("add zip dir");
    }
    for (name, bytes) in entries {
        zip.start_file(name, options).expect("start zip entry");
        zip.write_all(bytes).expect("write zip entry");
    }
    zip.finish().expect("finish zip");
}

fn run_unzip(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let _guard = outputs
        .map(|count| crate::output_count::push_output_count(Some(count)))
        .transpose()
        .expect("output-count scope");
    block_on(unzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_unzip_http(args: Vec<Value>) -> BuiltinResult<Value> {
    let driver = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("test HTTP driver");
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_http(Rc::new(NativeHttpService::new(driver.handle().clone()))),
        );
    block_on(unzip_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

fn cell_strings(value: Value) -> Vec<String> {
    let Value::Cell(cell) = value else {
        panic!("expected cell array");
    };
    assert_eq!(cell.cols, 1);
    assert_eq!(cell.rows, cell.data().len());
    cell.data()
        .iter()
        .map(|value| {
            let Value::CharArray(chars) = value else {
                panic!("unzip paths must be character rows")
            };
            assert_eq!(chars.rows, 1);
            String::try_from(value).expect("cell character path")
        })
        .collect()
}

fn spawn_server<F>(handler: F) -> String
where
    F: FnOnce(TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            handler(stream);
        }
    });
    format!("http://{}", addr)
}

fn respond_with(mut stream: TcpStream, content_type: &str, body: &[u8]) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
        body.len(),
        content_type
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.write_all(body);
}

fn drain_request(stream: &mut TcpStream) {
    let mut buffer = [0u8; 256];
    while let Ok(read) = stream.read(&mut buffer) {
        if read == 0 || buffer[..read].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
}

#[test]
fn unzip_registers_public_descriptor() {
    assert_eq!(runmat_builtins::UNZIP_CATALOG_ENTRY.identity.name, "unzip");
    let labels = UNZIP_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert!(labels.contains(&"unzip(zipfilename)"));
    assert!(labels.contains(&"unzip(zipfilename, outputfolder)"));
    assert!(labels.contains(&"filenames = unzip(___)"));
}

#[test]
fn unzip_extracts_local_archive_to_output_folder() {
    let archive = unique_zip_path("unzip_local");
    write_test_zip(
        &archive,
        &[("data/a.txt", b"alpha"), ("data/nested/b.txt", b"beta")],
    );
    let out = tempdir().expect("output folder");

    let value = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        None,
    )
    .expect("unzip");

    let extracted = cell_strings(value);
    let expected_a = out
        .path()
        .join("data")
        .join("a.txt")
        .to_string_lossy()
        .to_string();
    let expected_b = out
        .path()
        .join("data")
        .join("nested")
        .join("b.txt")
        .to_string_lossy()
        .to_string();
    assert_eq!(extracted, vec![expected_a, expected_b]);
    assert_eq!(
        fs::read_to_string(out.path().join("data").join("a.txt")).unwrap(),
        "alpha"
    );
    assert_eq!(
        fs::read_to_string(out.path().join("data").join("nested").join("b.txt")).unwrap(),
        "beta"
    );
    let _ = fs::remove_file(archive);
}

#[test]
fn unzip_zero_requested_outputs_returns_empty_output_list() {
    let archive = unique_zip_path("unzip_zero_outputs");
    write_test_zip(&archive, &[("a.txt", b"alpha")]);
    let out = tempdir().expect("output folder");

    let value = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        Some(0),
    )
    .expect("unzip");

    assert_eq!(value, Value::output_list(Vec::new()));
    assert_eq!(
        fs::read_to_string(out.path().join("a.txt")).unwrap(),
        "alpha"
    );
    let _ = fs::remove_file(archive);
}

#[test]
fn unzip_rejects_too_many_outputs_before_extracting() {
    let archive = unique_zip_path("unzip_too_many_outputs");
    write_test_zip(&archive, &[("a.txt", b"alpha")]);
    let out = tempdir().expect("output folder");

    let err = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        Some(2),
    )
    .expect_err("too many outputs");

    assert_eq!(err.identifier(), Some("RunMat:unzip:OutputCount"));
    assert!(!out.path().join("a.txt").exists());
    let _ = fs::remove_file(archive);
}

#[test]
fn numeric_resident_path_is_rejected_without_provider_download_or_filesystem_effect() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let owner = Box::leak(Box::new(CountingProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _thread = ThreadProviderGuard::set(Some(owner));
    let resident = owner
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let root = tempdir().unwrap();
    let destination = root.path().join("never-created");
    let error = block_on(unzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::GpuTensor(resident),
            Value::String(destination.to_string_lossy().into_owned().into()),
        ]),
    ))
    .expect_err("numeric gpuArray cannot be a ZIP source");
    assert_eq!(error.identifier(), UNZIP_ERROR_FILENAME.identifier);
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 0);
    assert!(!destination.exists());
}

#[test]
fn paged_character_controls_reject_at_public_dispatch_before_effects() {
    let root = tempdir().expect("output fixture");
    let destination = root.path().join("never-created");
    let paged = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .expect("real paged character value"),
    );
    let source_error = run_unzip(
        vec![
            paged.clone(),
            Value::String(destination.to_string_lossy().into_owned().into()),
        ],
        Some(1),
    )
    .expect_err("paged source must be rejected before filesystem access");
    assert_eq!(source_error.identifier(), UNZIP_ERROR_FILENAME.identifier);
    let destination_error = run_unzip(
        vec![
            Value::String("https://example.invalid/archive.zip".into()),
            paged,
        ],
        Some(1),
    )
    .expect_err("paged destination must be rejected before HTTP access");
    assert_eq!(
        destination_error.identifier(),
        UNZIP_ERROR_OUTPUT_FOLDER.identifier
    );
    assert!(!destination.exists());
}

#[test]
fn cancelled_runtime_context_stops_before_destination_creation() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let root = tempdir().unwrap();
    let destination = root.path().join("never-created");
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let error = block_on(unzip_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String(
                root.path()
                    .join("missing.zip")
                    .to_string_lossy()
                    .into_owned()
                    .into(),
            ),
            Value::String(destination.to_string_lossy().into_owned().into()),
        ]),
    ))
    .expect_err("cancelled call");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(!destination.exists());
}

#[test]
fn unzip_rejects_parent_traversal_entries() {
    let archive = unique_zip_path("unzip_traversal");
    write_test_zip(&archive, &[("../escape.txt", b"bad")]);
    let out = tempdir().expect("output folder");

    let err = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        None,
    )
    .expect_err("unsafe entry");

    assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
    assert!(!out.path().join("escape.txt").exists());
    let _ = fs::remove_file(archive);
}

#[test]
fn unzip_rejects_absolute_drive_and_backslash_traversal_entries() {
    for entry_name in ["/abs.txt", "C:/abs.txt", "nested\\..\\escape.txt"] {
        let archive = unique_zip_path("unzip_unsafe_name");
        write_test_zip(&archive, &[(entry_name, b"bad")]);
        let out = tempdir().expect("output folder");

        let err = run_unzip(
            vec![
                Value::from(archive.to_string_lossy().to_string()),
                Value::from(out.path().to_string_lossy().to_string()),
            ],
            None,
        )
        .expect_err("unsafe entry");

        assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
        let _ = fs::remove_file(archive);
    }
}

#[cfg(unix)]
#[test]
fn unzip_rejects_existing_symlink_components() {
    let archive = unique_zip_path("unzip_symlink_component");
    write_test_zip(&archive, &[("link/escape.txt", b"bad")]);
    let out = tempdir().expect("output folder");
    let outside = tempdir().expect("outside folder");
    std::os::unix::fs::symlink(outside.path(), out.path().join("link")).expect("symlink");

    let err = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        None,
    )
    .expect_err("symlink component");

    assert_eq!(err.identifier(), Some("RunMat:unzip:UnsafePath"));
    assert!(!outside.path().join("escape.txt").exists());
    let _ = fs::remove_file(archive);
}

#[test]
fn unzip_directory_only_archive_returns_empty_cell() {
    let archive = unique_zip_path("unzip_dir_only");
    write_test_zip_with_dirs(&archive, &["folder/"], &[]);
    let out = tempdir().expect("output folder");

    let value = run_unzip(
        vec![
            Value::from(archive.to_string_lossy().to_string()),
            Value::from(out.path().to_string_lossy().to_string()),
        ],
        None,
    )
    .expect("unzip");

    let extracted = cell_strings(value);
    assert!(extracted.is_empty());
    assert!(out.path().join("folder").is_dir());
    let _ = fs::remove_file(archive);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn unzip_downloads_url_archive() {
    let archive = unique_zip_path("unzip_url");
    write_test_zip(&archive, &[("remote.txt", b"downloaded")]);
    let bytes = fs::read(&archive).expect("read archive");
    let url = spawn_server(move |mut stream| {
        drain_request(&mut stream);
        respond_with(stream, "application/zip", &bytes);
    });
    let out = tempdir().expect("output folder");

    run_unzip_http(vec![
        Value::from(url),
        Value::from(out.path().to_string_lossy().to_string()),
    ])
    .expect("unzip URL");

    assert_eq!(
        fs::read_to_string(out.path().join("remote.txt")).unwrap(),
        "downloaded"
    );
    let _ = fs::remove_file(archive);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn unzip_http_status_errors() {
    let url = spawn_server(move |mut stream| {
        drain_request(&mut stream);
        let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(response.as_bytes());
    });
    let out = tempdir().expect("output parent");
    let destination = out.path().join("not-created");

    let err = run_unzip_http(vec![
        Value::from(url),
        Value::from(destination.to_string_lossy().to_string()),
    ])
    .expect_err("http status");

    assert_eq!(err.identifier(), Some("RunMat:unzip:Http"));
    assert!(!destination.exists());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn unzip_rejects_oversized_http_head_before_destination_creation() {
    let url = spawn_server(move |mut stream| {
        drain_request(&mut stream);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_COMPRESSED_ARCHIVE_BYTES + 1
        );
        let _ = stream.write_all(response.as_bytes());
    });
    let out = tempdir().expect("output parent");
    let destination = out.path().join("not-created");
    let error = run_unzip_http(vec![
        Value::from(url),
        Value::from(destination.to_string_lossy().to_string()),
    ])
    .expect_err("bounded response head");
    assert_eq!(error.identifier(), UNZIP_ERROR_HTTP.identifier);
    assert!(!destination.exists());
}

#[test]
fn unpolled_url_invocation_does_not_create_destination() {
    let out = tempdir().expect("output parent");
    let destination = out.path().join("not-created");
    let future = unzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("http://127.0.0.1:1/archive.zip"),
            Value::from(destination.to_string_lossy().to_string()),
        ]),
    );
    drop(future);
    assert!(!destination.exists());
}

#[test]
fn compressed_size_limit_is_enforced() {
    let err = ensure_compressed_size_within_limit(MAX_COMPRESSED_ARCHIVE_BYTES + 1)
        .expect_err("compressed limit");
    assert_eq!(err.identifier(), Some("RunMat:unzip:Http"));
}
