use std::io;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::context::RuntimeServicePorts;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::FsProvider;
use runmat_value::Value;

use super::support::{
    encoded, runtime, AmbientSwapRestoration, RecordingProvider, ReplacingHttpService,
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn public_gzip_keeps_captured_owner_when_ambient_provider_is_replaced() {
    let _lock = runmat_filesystem::provider_override_lock();
    let replacement = Arc::new(RecordingProvider::new("/replacement"));
    replacement
        .memory
        .write_project_path("/selected/source.bin", b"wrong")
        .expect("replacement trap");
    let swap = Arc::new(Mutex::new(None));
    let replacement_owner: Arc<dyn FsProvider> = replacement.clone();
    let selected = Arc::new(
        RecordingProvider::new("/selected")
            .with_replacement_on_read(replacement_owner, Arc::clone(&swap)),
    );
    selected
        .memory
        .write_project_path("/selected/source.bin", b"selected")
        .expect("selected source");
    let selected_scope = runmat_filesystem::replace_provider(selected.clone());
    let _restore_swap = AmbientSwapRestoration::new(&selected_scope, Arc::clone(&swap));
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");

    let value = futures::executor::block_on(super::super::super::gzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("source.bin".into()),
            Value::String("out".into()),
        ]),
    ))
    .expect("public gzip through captured owner");

    let Value::Cell(outputs) = value else {
        panic!("gzip output must be a cell")
    };
    let expected = std::path::Path::new("out").join("source.bin.gz");
    assert!(matches!(
        outputs.data(),
        [Value::CharArray(path)] if runmat_value::decode_utf16_code_units(path.data()).expect("valid UTF-16 fixture") == expected.to_str().expect("UTF-8 fixture path")
    ));
    assert!(selected
        .memory
        .read_project_path("/selected/out/source.bin.gz")
        .is_ok());
    assert_eq!(selected.reads.load(Ordering::SeqCst), 1);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 1);
    assert_eq!(replacement.reads.load(Ordering::SeqCst), 0);
    assert_eq!(replacement.writes.load(Ordering::SeqCst), 0);
    assert!(replacement
        .memory
        .read_project_path("/selected/out/source.bin.gz")
        .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn public_url_gunzip_keeps_filesystem_owner_captured_before_http() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected"));
    let replacement = Arc::new(RecordingProvider::new("/replacement"));
    let swap = Arc::new(Mutex::new(None));
    let sends = Arc::new(AtomicUsize::new(0));
    let replacement_owner: Arc<dyn FsProvider> = replacement.clone();
    let cancellation = CancellationSource::new();
    let runtime = runtime(&cancellation).with_service_ports(
        RuntimeServicePorts::default().with_http(Rc::new(ReplacingHttpService {
            body: encoded(b"selected response"),
            sends: Arc::clone(&sends),
            replacement: replacement_owner,
            guard: Arc::clone(&swap),
        })),
    );
    let selected_scope = runmat_filesystem::replace_provider(selected.clone());
    let _restore_swap = AmbientSwapRestoration::new(&selected_scope, Arc::clone(&swap));
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");

    let value = futures::executor::block_on(super::super::super::gunzip_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("https://example.invalid/archive.bin.gz".into()),
            Value::String("out".into()),
        ]),
    ))
    .expect("public URL gunzip");

    let Value::Cell(outputs) = value else {
        panic!("gunzip output must be a cell")
    };
    let expected = std::path::Path::new("out").join("archive.bin");
    assert!(matches!(
        outputs.data(),
        [Value::CharArray(path)] if runmat_value::decode_utf16_code_units(path.data()).expect("valid UTF-16 fixture") == expected.to_str().expect("UTF-8 fixture path")
    ));
    assert_eq!(sends.load(Ordering::SeqCst), 1);
    assert_eq!(
        selected
            .memory
            .read_project_path("/selected/out/archive.bin")
            .expect("selected output"),
        b"selected response"
    );
    assert_eq!(selected.writes.load(Ordering::SeqCst), 1);
    assert_eq!(replacement.writes.load(Ordering::SeqCst), 0);
    assert!(replacement
        .memory
        .read_project_path("/selected/out/archive.bin")
        .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn read_revocation_stops_codec_and_output_effects_with_typed_source() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected").with_read_revocation());
    selected
        .memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");

    let error = futures::executor::block_on(super::super::super::gzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("source.bin".into()),
            Value::String("out".into()),
        ]),
    ))
    .expect_err("revoked owner");

    assert_eq!(selected.reads.load(Ordering::SeqCst), 1);
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
fn provider_error_remains_primary_when_the_same_failed_read_revokes_its_lease() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(
        RecordingProvider::new("/selected").with_read_error_and_revocation(io::ErrorKind::NotFound),
    );
    selected
        .memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");

    let error = futures::executor::block_on(super::super::super::gzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("source.bin".into()),
            Value::String("out".into()),
        ]),
    ))
    .expect_err("selected provider error");

    assert_eq!(selected.reads.load(Ordering::SeqCst), 1);
    assert_eq!(selected.creates.load(Ordering::SeqCst), 0);
    assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
    assert!(error.source.as_deref().is_some_and(|source| {
        source
            .downcast_ref::<io::Error>()
            .is_some_and(|error| error.kind() == io::ErrorKind::NotFound)
    }));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn wildcard_enumeration_stops_between_provider_awaits_when_authority_is_revoked() {
    let _lock = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RecordingProvider::new("/selected").with_read_dir_revocation());
    selected
        .memory
        .write_project_path("/selected/folder/source.bin", b"source")
        .expect("source");
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir("/selected").expect("selected cwd");

    let error = futures::executor::block_on(super::super::super::gzip_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("folder/*.bin".into()),
            Value::String("out".into()),
        ]),
    ))
    .expect_err("revoked enumeration owner");

    assert_eq!(selected.read_dirs.load(Ordering::SeqCst), 1);
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
fn revoked_metadata_errors_remain_primary_and_never_start_a_fallback_operation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let cases = [
        (
            "/selected/folder",
            2,
            io::ErrorKind::InvalidData,
            1,
            "directory entry metadata fallback",
        ),
        (
            "/selected/folder/*.bin",
            1,
            io::ErrorKind::NotFound,
            0,
            "matching directory normalization",
        ),
        (
            "/selected/folder/*.bin",
            3,
            io::ErrorKind::Other,
            1,
            "matched result metadata fallback",
        ),
    ];

    for (input, failure_at, kind, expected_read_dirs, label) in cases {
        let selected = Arc::new(
            RecordingProvider::new("/selected")
                .with_metadata_error_and_revocation(failure_at, kind),
        );
        selected
            .memory
            .write_project_path("/selected/folder/source.bin", b"source")
            .expect("source");
        let _selected = runmat_filesystem::replace_provider(selected.clone());

        let error = futures::executor::block_on(super::super::super::gzip_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(input.into()),
                Value::String("/selected/out".into()),
            ]),
        ))
        .expect_err(label);

        assert_eq!(selected.metadata_calls.load(Ordering::SeqCst), failure_at);
        assert_eq!(
            selected.read_dirs.load(Ordering::SeqCst),
            expected_read_dirs,
            "{label}"
        );
        assert_eq!(
            selected.symlink_metadata_calls.load(Ordering::SeqCst),
            0,
            "{label}"
        );
        assert_eq!(selected.reads.load(Ordering::SeqCst), 0, "{label}");
        assert_eq!(selected.creates.load(Ordering::SeqCst), 0, "{label}");
        assert_eq!(selected.writes.load(Ordering::SeqCst), 0, "{label}");
        assert!(error.source.as_deref().is_some_and(|source| {
            source
                .downcast_ref::<io::Error>()
                .is_some_and(|error| error.kind() == kind)
        }));
    }
}
