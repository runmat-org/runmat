use std::io;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::context::RuntimeServicePorts;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::FsProvider;
use runmat_value::Value;

use super::support::{encoded, runtime, RecordingProvider, ReplacingHttpService};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn public_call_rejects_an_initially_invalid_resource_lease_before_provider_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected"));
    selected.valid.store(false, Ordering::SeqCst);
    let _selected = runmat_filesystem::replace_provider(selected.clone());

    let error = futures::executor::block_on(super::super::super::gzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "source.bin".into(),
        )]),
    ))
    .expect_err("invalid resource lease");

    assert_eq!(selected.metadata_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.read_dirs.load(Ordering::SeqCst), 0);
    assert_eq!(selected.reads.load(Ordering::SeqCst), 0);
    assert_eq!(selected.creates.load(Ordering::SeqCst), 0);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
    assert!(error.source.as_deref().is_some_and(|source| {
        source
            .downcast_ref::<io::Error>()
            .is_some_and(|error| error.kind() == io::ErrorKind::PermissionDenied)
    }));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_local_gunzip_filename_rejects_before_provider_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected"));
    let error = runmat_filesystem::with_provider_override(selected.clone(), || {
        futures::executor::block_on(super::super::super::gunzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "/".into(),
            )]),
        ))
        .expect_err("root has no archive filename")
    });

    assert!(error.message().contains("invalid filename"), "{error}");
    assert_eq!(selected.reads.load(Ordering::SeqCst), 0);
    assert_eq!(selected.creates.load(Ordering::SeqCst), 0);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_url_gunzip_filename_rejects_before_http_or_filesystem_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected"));
    let replacement = Arc::new(RecordingProvider::new("/replacement"));
    let sends = Arc::new(AtomicUsize::new(0));
    let replacement_owner: Arc<dyn FsProvider> = replacement;
    let cancellation = CancellationSource::new();
    let runtime = runtime(&cancellation).with_service_ports(
        RuntimeServicePorts::default().with_http(Rc::new(ReplacingHttpService {
            body: encoded(b"unused"),
            sends: Arc::clone(&sends),
            replacement: replacement_owner,
            guard: Arc::new(Mutex::new(None)),
        })),
    );
    let _selected = runmat_filesystem::replace_provider(selected.clone());

    let error = futures::executor::block_on(super::super::super::gunzip_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "https://example.invalid/.gz".into(),
        )]),
    ))
    .expect_err("empty decompressed filename");

    assert!(
        error.message().contains("invalid output filename"),
        "{error}"
    );
    assert_eq!(sends.load(Ordering::SeqCst), 0);
    assert_eq!(selected.metadata_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.reads.load(Ordering::SeqCst), 0);
    assert_eq!(selected.creates.load(Ordering::SeqCst), 0);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
}
