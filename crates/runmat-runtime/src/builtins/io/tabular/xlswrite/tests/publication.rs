//! Public status forms must not imply rollback after uncertain publication.

use super::*;
use async_trait::async_trait;
use futures::task::{waker_ref, ArcWake};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationDisposition, FsPublicationFailure,
    FsResourceLease, FsStagedOutput, MemoryFsProvider, OpenFlags,
};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;

#[derive(Default)]
pub(super) struct LifecycleEvidence {
    pub(super) begins: AtomicUsize,
    pub(super) resource_probes: AtomicUsize,
    writes: AtomicUsize,
    publish_polls: AtomicUsize,
    retired_stages: AtomicUsize,
}

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl ArcWake for WakeCount {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
pub(super) enum LifecycleBehavior {
    Pending(Arc<LifecycleEvidence>),
    ReadyCancel(CancellationSource, Arc<LifecycleEvidence>),
}

impl LifecycleBehavior {
    fn evidence(&self) -> &Arc<LifecycleEvidence> {
        match self {
            Self::Pending(evidence) | Self::ReadyCancel(_, evidence) => evidence,
        }
    }
}

struct PublicationProvider {
    memory: MemoryFsProvider,
    disposition: FsPublicationDisposition,
    lifecycle: Option<LifecycleBehavior>,
}

struct PublicationStage {
    memory: MemoryFsProvider,
    destination: PathBuf,
    disposition: FsPublicationDisposition,
    lifecycle: Option<LifecycleBehavior>,
    bytes: Vec<u8>,
}

impl Drop for PublicationStage {
    fn drop(&mut self) {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle
                .evidence()
                .retired_stages
                .fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[async_trait(?Send)]
impl FsStagedOutput for PublicationStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle.evidence().writes.fetch_add(1, Ordering::SeqCst);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }

    async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle
                .evidence()
                .publish_polls
                .fetch_add(1, Ordering::SeqCst);
            match lifecycle {
                LifecycleBehavior::Pending(_) => {
                    futures::future::pending::<()>().await;
                    unreachable!("pending publication fixture")
                }
                LifecycleBehavior::ReadyCancel(source, _) => {
                    self.memory
                        .write_project_path(&self.destination, &self.bytes)
                        .map_err(FsPublicationFailure::not_published)?;
                    source.cancel(CancellationReason::User);
                    return Ok(());
                }
            }
        }
        if self.disposition != FsPublicationDisposition::NotPublished {
            self.memory
                .write_project_path(&self.destination, &self.bytes)
                .map_err(FsPublicationFailure::not_published)?;
        }
        Err(FsPublicationFailure::new(
            self.disposition,
            io::Error::other("injected publication failure"),
        ))
    }
}

#[async_trait(?Send)]
impl FsProvider for PublicationProvider {
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle
                .evidence()
                .resource_probes
                .fetch_add(1, Ordering::SeqCst);
        }
        None
    }

    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        self.memory.current_dir_override()
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        if let Some(lifecycle) = &self.lifecycle {
            lifecycle.evidence().begins.fetch_add(1, Ordering::SeqCst);
        }
        Ok(Some(Box::new(PublicationStage {
            memory: self.memory.clone(),
            destination: destination.to_path_buf(),
            disposition: self.disposition,
            lifecycle: self.lifecycle.clone(),
            bytes: Vec::new(),
        })))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.memory.write(path, data).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.memory.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.memory.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

fn fixture_root() -> &'static str {
    "/publication"
}

fn public_call(
    disposition: FsPublicationDisposition,
    output_count: usize,
) -> (BuiltinResult<Value>, MemoryFsProvider) {
    let memory = MemoryFsProvider::with_current_dir(fixture_root());
    block_on(memory.create_dir_all(Path::new(fixture_root()))).expect("fixture directory");
    let provider = Arc::new(PublicationProvider {
        memory: memory.clone(),
        disposition,
        lifecycle: None,
    });
    let _restore = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(fixture_root()).expect("fixture cwd");
    let _outputs =
        crate::output_count::push_output_count(Some(output_count)).expect("output-count scope");
    let result = block_on(xlswrite_call(
        Value::from("out.xlsx"),
        Value::Num(7.0),
        Vec::new(),
    ));
    (result, memory)
}

pub(super) fn with_lifecycle(
    lifecycle: LifecycleBehavior,
    output_count: usize,
    test: impl FnOnce(MemoryFsProvider, Arc<LifecycleEvidence>),
) {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir(fixture_root());
    block_on(memory.create_dir_all(Path::new(fixture_root()))).expect("fixture directory");
    let evidence = lifecycle.evidence().clone();
    let provider = Arc::new(PublicationProvider {
        memory: memory.clone(),
        disposition: FsPublicationDisposition::NotPublished,
        lifecycle: Some(lifecycle),
    });
    let _restore = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(fixture_root()).expect("fixture cwd");
    let _outputs =
        crate::output_count::push_output_count(Some(output_count)).expect("output-count scope");
    test(memory, evidence);
}

fn destination_exists(memory: &MemoryFsProvider) -> bool {
    memory
        .read_project_path(Path::new(fixture_root()).join("out.xlsx"))
        .is_ok()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_call_drop_during_pending_publish_retires_stage_without_replay() {
    for output_count in [1, 2] {
        let evidence = Arc::new(LifecycleEvidence::default());
        with_lifecycle(
            LifecycleBehavior::Pending(evidence),
            output_count,
            |memory, evidence| {
                let cancellation = CancellationSource::new();
                let runtime = RuntimeContext::with_cancellation(
                    Rc::new(RuntimeExecutionService::new()),
                    cancellation.token(),
                );
                let mut call = Box::pin(xlswrite_builtin(
                    Some(runtime),
                    Value::from("out.xlsx"),
                    Value::Num(7.0),
                    crate::builtins::common::test_support::builtin_arguments(Vec::new()),
                ));
                let waker = futures::task::noop_waker();
                for _ in 0..8 {
                    assert!(matches!(
                        call.as_mut().poll(&mut Context::from_waker(&waker)),
                        Poll::Pending
                    ));
                    if evidence.publish_polls.load(Ordering::SeqCst) == 1 {
                        break;
                    }
                }
                assert_eq!(evidence.begins.load(Ordering::SeqCst), 1);
                assert!(evidence.writes.load(Ordering::SeqCst) > 0);
                assert_eq!(evidence.publish_polls.load(Ordering::SeqCst), 1);
                assert_eq!(evidence.retired_stages.load(Ordering::SeqCst), 0);
                drop(call);
                assert_eq!(evidence.retired_stages.load(Ordering::SeqCst), 1);
                assert_eq!(evidence.begins.load(Ordering::SeqCst), 1);
                assert!(!destination_exists(&memory));
            },
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_call_cancel_during_pending_publish_is_indeterminate_not_false() {
    let evidence = Arc::new(LifecycleEvidence::default());
    let source = CancellationSource::new();
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        2,
        |memory, evidence| {
            let runtime = RuntimeContext::with_cancellation(
                Rc::new(RuntimeExecutionService::new()),
                source.token(),
            );
            let mut call = Box::pin(xlswrite_builtin(
                Some(runtime),
                Value::from("out.xlsx"),
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ));
            let wakes = Arc::new(WakeCount::default());
            let waker = waker_ref(&wakes);
            for _ in 0..8 {
                assert!(matches!(
                    call.as_mut().poll(&mut Context::from_waker(&waker)),
                    Poll::Pending
                ));
                if evidence.publish_polls.load(Ordering::SeqCst) == 1 {
                    break;
                }
            }
            assert_eq!(evidence.publish_polls.load(Ordering::SeqCst), 1);
            let wakes_before_cancel = wakes.0.load(Ordering::SeqCst);
            source.cancel(CancellationReason::User);
            assert!(
                wakes.0.load(Ordering::SeqCst) > wakes_before_cancel,
                "cancellation must wake the pending public call"
            );
            let Poll::Ready(result) = call.as_mut().poll(&mut Context::from_waker(&waker)) else {
                panic!("woken publication call must complete without another external event")
            };
            let error = result.expect_err("uncertain publication cannot become false status");
            assert_eq!(
                write_owner::publication_error_disposition(&error),
                Some(FsPublicationDisposition::Indeterminate)
            );
            assert!(is_cancelled(&error));
            assert_eq!(evidence.begins.load(Ordering::SeqCst), 1);
            assert_eq!(evidence.publish_polls.load(Ordering::SeqCst), 1);
            assert_eq!(evidence.retired_stages.load(Ordering::SeqCst), 1);
            assert!(!destination_exists(&memory));
        },
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dispatcher_binding_preserves_two_output_known_unpublished_status() {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir(fixture_root());
    block_on(memory.create_dir_all(Path::new(fixture_root()))).expect("fixture directory");
    let provider = Arc::new(PublicationProvider {
        memory: memory.clone(),
        disposition: FsPublicationDisposition::NotPublished,
        lifecycle: None,
    });
    let _restore = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(fixture_root()).expect("fixture cwd");
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let outputs = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "xlswrite",
        &[Value::from("out.xlsx"), Value::Num(7.0)],
        2,
    ))
    .expect("public dispatcher must select xlswrite");
    let values = output_list(outputs.value());
    assert_eq!(values.len(), 2);
    assert_eq!(values[0], Value::Bool(false));
    let Value::Struct(message) = &values[1] else {
        panic!("message struct")
    };
    assert!(
        matches!(message.fields.get("identifier"), Some(Value::String(id)) if id == "RunMat:xlswrite:Io")
    );
    assert!(!destination_exists(&memory));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_call_ready_publish_then_cancel_keeps_published_disposition() {
    let evidence = Arc::new(LifecycleEvidence::default());
    let source = CancellationSource::new();
    with_lifecycle(
        LifecycleBehavior::ReadyCancel(source.clone(), evidence),
        1,
        |memory, evidence| {
            let runtime = RuntimeContext::with_cancellation(
                Rc::new(RuntimeExecutionService::new()),
                source.token(),
            );
            let error = block_on(xlswrite_builtin(
                Some(runtime),
                Value::from("out.xlsx"),
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("completed publication with cancellation is not false status");
            assert_eq!(
                write_owner::publication_error_disposition(&error),
                Some(FsPublicationDisposition::Published)
            );
            assert!(destination_exists(&memory));
            assert_eq!(evidence.begins.load(Ordering::SeqCst), 1);
            assert_eq!(evidence.publish_polls.load(Ordering::SeqCst), 1);
            assert_eq!(evidence.retired_stages.load(Ordering::SeqCst), 1);
        },
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_call_precancelled_before_first_poll_has_no_provider_effect() {
    let evidence = Arc::new(LifecycleEvidence::default());
    let source = CancellationSource::new();
    with_lifecycle(
        LifecycleBehavior::Pending(evidence),
        2,
        |memory, evidence| {
            let runtime = RuntimeContext::with_cancellation(
                Rc::new(RuntimeExecutionService::new()),
                source.token(),
            );
            let call = xlswrite_builtin(
                Some(runtime),
                Value::from("out.xlsx"),
                Value::Num(7.0),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            );
            source.cancel(CancellationReason::User);
            let error = block_on(call).expect_err("pre-cancelled public call must not publish");
            assert!(is_cancelled(&error));
            assert_eq!(evidence.begins.load(Ordering::SeqCst), 0);
            assert_eq!(evidence.writes.load(Ordering::SeqCst), 0);
            assert_eq!(evidence.publish_polls.load(Ordering::SeqCst), 0);
            assert_eq!(evidence.retired_stages.load(Ordering::SeqCst), 0);
            assert!(!destination_exists(&memory));
        },
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_status_forms_only_capture_known_unpublished_failure() {
    let _lock = runmat_filesystem::provider_override_lock();
    for outputs in [1, 2] {
        let (result, memory) = public_call(FsPublicationDisposition::NotPublished, outputs);
        let status = result.expect("known unpublished failure becomes status");
        let values = output_list(&status);
        assert_eq!(values[0], Value::Bool(false));
        assert_eq!(values.len(), outputs);
        if outputs == 2 {
            let Value::Struct(message) = &values[1] else {
                panic!("message struct")
            };
            assert!(
                matches!(message.fields.get("identifier"), Some(Value::String(id)) if id == "RunMat:xlswrite:Io")
            );
        }
        assert!(memory
            .read_project_path(Path::new(fixture_root()).join("out.xlsx"))
            .is_err());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn public_status_forms_propagate_published_and_indeterminate_failures() {
    let _lock = runmat_filesystem::provider_override_lock();
    for disposition in [
        FsPublicationDisposition::Published,
        FsPublicationDisposition::Indeterminate,
    ] {
        for outputs in [1, 2] {
            let (result, memory) = public_call(disposition, outputs);
            let error = result.expect_err("uncertain or committed publication must throw");
            assert_eq!(
                write_owner::publication_error_disposition(&error),
                Some(disposition)
            );
            assert!(memory
                .read_project_path(Path::new(fixture_root()).join("out.xlsx"))
                .is_ok());
        }
    }
}
