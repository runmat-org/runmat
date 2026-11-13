//! End-to-end audio calls suspended by the selected filesystem owner.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::task::noop_waker_ref;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll};

#[derive(Default)]
struct Evidence {
    attempts: AtomicUsize,
    completed: AtomicUsize,
    retired_futures: AtomicUsize,
    retired_handles: AtomicUsize,
}

struct RetireFuture(Arc<Evidence>);
impl Drop for RetireFuture {
    fn drop(&mut self) {
        self.0.retired_futures.fetch_add(1, Ordering::SeqCst);
    }
}

struct CountedHandle {
    inner: Box<dyn FileHandle>,
    evidence: Arc<Evidence>,
}
impl Drop for CountedHandle {
    fn drop(&mut self) {
        self.evidence.retired_handles.fetch_add(1, Ordering::SeqCst);
    }
}
impl Read for CountedHandle {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.inner.read(bytes)
    }
}
impl Write for CountedHandle {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
impl Seek for CountedHandle {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}
#[async_trait(?Send)]
impl FileHandle for CountedHandle {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        self.inner.metadata_async().await
    }
}

struct GatedProvider {
    memory: MemoryFsProvider,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    evidence: Arc<Evidence>,
}

#[async_trait(?Send)]
impl FsProvider for GatedProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/engine"))
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.evidence.attempts.fetch_add(1, Ordering::SeqCst);
        let _retirement = RetireFuture(self.evidence.clone());
        let receiver = self.gate.lock().expect("gate lock").take();
        if let Some(receiver) = receiver {
            receiver
                .await
                .map_err(|_| io::Error::other("audio gate was abandoned"))?;
        }
        let inner = self.memory.open(path, flags)?;
        self.evidence.completed.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(CountedHandle {
            inner,
            evidence: self.evidence.clone(),
        }))
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

fn fixture(
    test: impl FnOnce(&mut RunMatSession, &crate::ExecutableUnit, Arc<Evidence>, oneshot::Sender<()>),
) {
    let _lock = runmat_filesystem::provider_override_lock();
    let memory = MemoryFsProvider::with_current_dir("/engine");
    memory
        .write_project_path("/engine/pcm.wav", &pcm16())
        .unwrap();
    let (sender, receiver) = oneshot::channel();
    let evidence = Arc::new(Evidence::default());
    let provider = Arc::new(GatedProvider {
        memory,
        gate: Mutex::new(Some(receiver)),
        evidence: evidence.clone(),
    });
    let _owner = replace_provider(provider);
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("audio-gated@1", "engineReadPcm.m", SOURCE),
            None,
        ))
        .unwrap();
        test(&mut session, &unit, evidence, sender);
    });
}

fn reset(session: &mut RunMatSession, unit: &crate::ExecutableUnit) {
    block_on(session.invoke_executable(
        unit,
        ProcedureInvocation {
            target: ProcedureTarget::Function("engineResetCount".into()),
            arguments: crate::ProcedureArguments::Source(Vec::new()),
            requested_outputs: 0,
        },
        &InvocationControl::default().force_generic_native(),
    ))
    .unwrap();
}

fn count(session: &mut RunMatSession, unit: &crate::ExecutableUnit) -> Value {
    let value = block_on(session.invoke_executable(
        unit,
        ProcedureInvocation {
            target: ProcedureTarget::Function("engineGetCount".into()),
            arguments: crate::ProcedureArguments::Source(Vec::new()),
            requested_outputs: 1,
        },
        &InvocationControl::default().force_generic_native(),
    ))
    .unwrap();
    value.as_slice()[0].clone()
}

fn poll_to_gate<F: Future>(future: std::pin::Pin<&mut F>, evidence: &Evidence) {
    let mut future = future;
    for _ in 0..8 {
        let mut context = Context::from_waker(noop_waker_ref());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        if evidence.attempts.load(Ordering::SeqCst) == 1 {
            return;
        }
    }
    panic!("audio invocation did not suspend at the provider open");
}

#[test]
fn native_audio_pending_open_resumes_without_replaying_source_effects() {
    fixture(|session, unit, evidence, release| {
        reset(session, unit);
        let control = InvocationControl::default().force_generic_native();
        let mut pending = Box::pin(session.invoke_executable(
            unit,
            invocation("engineReadNoReplay", "/engine/pcm.wav", 2),
            &control,
        ));
        poll_to_gate(pending.as_mut(), &evidence);
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        let result = block_on(pending).unwrap();
        let [Value::Num(effect_count), Value::Tensor(samples)] = result.as_slice() else {
            panic!("expected effect count and native PCM16 samples");
        };
        assert_eq!(*effect_count, 1.0);
        assert_eq!(samples.shape(), &[2, 2]);
        assert_eq!(
            samples.integer_storage(),
            Some(&IntegerStorage::I16(vec![-4, 10, 5, -10]))
        );
        assert_eq!(evidence.attempts.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_futures.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 1);
        assert_eq!(count(session, unit), Value::Num(1.0));
    });
}

#[test]
fn native_audioinfo_pending_open_resumes_once_with_exact_metadata() {
    fixture(|session, unit, evidence, release| {
        reset(session, unit);
        let control = InvocationControl::default().force_generic_native();
        let mut pending = Box::pin(session.invoke_executable(
            unit,
            invocation("engineInfoNoReplay", "/engine/pcm.wav", 2),
            &control,
        ));
        poll_to_gate(pending.as_mut(), &evidence);
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);
        release.send(()).unwrap();
        let result = block_on(pending).unwrap();
        let [Value::Num(effect_count), Value::Struct(info)] = result.as_slice() else {
            panic!("expected count and audioinfo struct");
        };
        assert_eq!(*effect_count, 1.0);
        assert_eq!(
            info.fields.get("Format"),
            Some(&Value::String("WAV".into()))
        );
        assert_eq!(info.fields.get("TotalSamples"), Some(&Value::Num(3.0)));
        assert_eq!(info.fields.get("NumChannels"), Some(&Value::Num(2.0)));
        assert_eq!(info.fields.get("SampleRate"), Some(&Value::Num(22_050.0)));
        assert_eq!(info.fields.get("BitsPerSample"), Some(&Value::Num(16.0)));
        assert_eq!(evidence.attempts.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_futures.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 1);
        assert_eq!(count(session, unit), Value::Num(1.0));
    });
}

#[test]
fn native_audio_pending_open_cancellation_retires_owner_and_does_not_replay() {
    fixture(|session, unit, evidence, release| {
        reset(session, unit);
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let control = InvocationControl::default()
            .force_generic_native()
            .with_cancellation(cancellation.token());
        let mut pending = Box::pin(session.invoke_executable(
            unit,
            invocation("engineReadNoReplay", "/engine/pcm.wav", 2),
            &control,
        ));
        poll_to_gate(pending.as_mut(), &evidence);
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(pending).expect_err("pending audio cancellation");
        let RunError::Runtime(error) = error else {
            panic!("expected typed cancellation");
        };
        assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(release.send(()).is_err(), "open future was not retired");
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);
        assert_eq!(evidence.retired_futures.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 0);
        assert_eq!(count(session, unit), Value::Num(1.0));
        let resumed = block_on(session.invoke_executable(
            unit,
            invocation("engineReadNoReplay", "/engine/pcm.wav", 2),
            &InvocationControl::default().force_generic_native(),
        ))
        .unwrap();
        assert_eq!(resumed.as_slice()[0], Value::Num(2.0));
        assert_eq!(evidence.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn native_audio_dropped_pending_invocation_retires_owner_without_replay() {
    fixture(|session, unit, evidence, release| {
        reset(session, unit);
        let control = InvocationControl::default().force_generic_native();
        let mut pending = Box::pin(session.invoke_executable(
            unit,
            invocation("engineReadNoReplay", "/engine/pcm.wav", 2),
            &control,
        ));
        poll_to_gate(pending.as_mut(), &evidence);
        drop(pending);
        assert!(
            release.send(()).is_err(),
            "dropped invocation kept open alive"
        );
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);
        assert_eq!(evidence.retired_futures.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 0);
        assert_eq!(count(session, unit), Value::Num(1.0));
        let fresh = block_on(session.invoke_executable(
            unit,
            invocation("engineReadNoReplay", "/engine/pcm.wav", 2),
            &InvocationControl::default().force_generic_native(),
        ))
        .unwrap();
        assert_eq!(fresh.as_slice()[0], Value::Num(2.0));
        assert_eq!(evidence.attempts.load(Ordering::SeqCst), 2);
        assert_eq!(evidence.retired_handles.load(Ordering::SeqCst), 1);
    });
}
