use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::Poll;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::Value;

use super::support::{poll_once, runtime, RecordingProvider};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pending_provider_read_is_dropped_on_cancellation_or_abandonment_without_output_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    for cancel in [false, true] {
        let selected = Arc::new(RecordingProvider::new("/selected").with_pending_reads());
        let cancellation = CancellationSource::new();
        let runtime = runtime(&cancellation);
        let mut owner = runmat_filesystem::with_provider_override(selected.clone(), || {
            super::super::ArchiveFilesystem::capture(
                super::super::super::codec::Operation::Compress,
                Some(&runtime),
            )
            .expect("capture owner")
        });
        let mut call = Box::pin(super::super::super::gzip_one(
            &mut owner,
            Path::new("source.bin"),
            Some(Path::new("out")),
        ));
        assert!(poll_once(call.as_mut()).is_pending());
        assert_eq!(selected.reads.load(Ordering::SeqCst), 1);

        if cancel {
            cancellation.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = poll_once(call.as_mut()) else {
                panic!("cancellation must wake a pending provider read")
            };
            assert_eq!(error.identifier(), Some("RunMat:parallel:Cancelled"));
        }
        drop(call);
        assert_eq!(selected.pending_dropped.load(Ordering::SeqCst), 1);
        assert_eq!(selected.creates.load(Ordering::SeqCst), 0);
        assert_eq!(selected.writes.load(Ordering::SeqCst), 0);
        assert!(selected
            .memory
            .read_project_path("/selected/out/source.bin.gz")
            .is_err());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn public_pending_write_is_dropped_on_cancellation_or_abandonment_without_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    for cancel in [false, true] {
        let selected = Arc::new(RecordingProvider::new("/selected").with_pending_writes());
        selected
            .memory
            .write_project_path("/selected/source.bin", b"source")
            .expect("source");
        let cancellation = CancellationSource::new();
        let runtime = runtime(&cancellation);
        let _selected = runmat_filesystem::replace_provider(selected.clone());
        runmat_filesystem::set_current_dir("/selected").expect("selected cwd");
        let args = vec![
            Value::String("source.bin".into()),
            Value::String("out".into()),
        ];
        let mut call = Box::pin(super::super::super::gzip_builtin(
            Some(runtime),
            crate::builtins::common::test_support::builtin_arguments(args),
        ));
        assert!(poll_once(call.as_mut()).is_pending());
        assert_eq!(selected.writes.load(Ordering::SeqCst), 1);

        if cancel {
            cancellation.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = poll_once(call.as_mut()) else {
                panic!("cancellation must wake a pending provider write")
            };
            assert_eq!(error.identifier(), Some("RunMat:parallel:Cancelled"));
        }
        drop(call);
        assert_eq!(selected.pending_write_dropped.load(Ordering::SeqCst), 1);
        assert!(selected
            .memory
            .read_project_path("/selected/out/source.bin.gz")
            .is_err());
    }
}
