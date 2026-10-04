use super::*;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, NativePathAuthority, NativePathGrant,
    NativePathIntent, NativeProcessRevocation, OpenFlags,
};
use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct RevocableDirectory {
    live: Arc<AtomicBool>,
    notifier: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
}

impl RevocableDirectory {
    fn new() -> Self {
        Self {
            live: Arc::new(AtomicBool::new(true)),
            notifier: Mutex::new(None),
        }
    }

    fn revoke(&self) {
        self.live.store(false, Ordering::SeqCst);
        if let Some(notifier) = self.notifier.lock().expect("notifier").take() {
            let _ = notifier.send(());
        }
    }
}

struct WatchLease {
    live: Arc<AtomicBool>,
    receiver: RefCell<Option<futures::channel::oneshot::Receiver<()>>>,
}

impl NativePathAuthority for WatchLease {
    fn is_valid(&self) -> bool {
        self.live.load(Ordering::SeqCst)
    }

    fn process_revocation(&self) -> NativeProcessRevocation<'_> {
        let receiver = self
            .receiver
            .borrow_mut()
            .take()
            .expect("one process watch");
        NativeProcessRevocation::Watch(Box::pin(async move {
            let _ = receiver.await;
        }))
    }
}

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "process test has no file I/O")
}

#[async_trait::async_trait(?Send)]
impl FsProvider for RevocableDirectory {
    fn acquire_native_path(
        &self,
        path: &Path,
        intent: NativePathIntent,
    ) -> io::Result<Option<NativePathGrant>> {
        assert_eq!(intent, NativePathIntent::ProcessWorkingDirectory);
        let (notifier, receiver) = futures::channel::oneshot::channel();
        *self.notifier.lock().expect("notifier") = Some(notifier);
        NativePathGrant::issued_by_provider(
            path.to_path_buf(),
            intent,
            Box::new(WatchLease {
                live: Arc::clone(&self.live),
                receiver: RefCell::new(Some(receiver)),
            }),
        )
        .map(Some)
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unused())
    }
}

struct DirectProcess;

impl crate::context::RuntimeProcessService for DirectProcess {
    fn output_complete(
        &self,
        command: runmat_process_host::HostCommand,
    ) -> crate::context::RuntimeServiceFuture<
        runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
    > {
        Box::pin(command.output_complete())
    }
    fn output_complete_with_input(
        &self,
        command: runmat_process_host::HostCommand,
        input: Vec<u8>,
    ) -> crate::context::RuntimeServiceFuture<
        runmat_process_host::ProcessHostResult<runmat_process_host::CompleteProcessOutput>,
    > {
        Box::pin(command.output_complete_with_input(input))
    }
}

#[test]
fn pre_revoked_directory_grant_rejects_shared_process_call_before_spawn() {
    let _serial = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RevocableDirectory::new());
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    selected.revoke();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_process(Rc::new(DirectProcess)),
        );
    let directory = tempfile::tempdir().expect("marker directory");
    let marker = directory.path().join("must-not-spawn");
    let mut child = runmat_process_host::HostCommand::new("/bin/sh");
    child.arguments = vec![
        "-c".into(),
        "printf started > \"$1\"".into(),
        "sh".into(),
        marker.to_string_lossy().into_owned(),
    ];
    let error = futures::executor::block_on(crate::context::process_invocation::output_complete(
        Some(&runtime),
        child,
        None,
    ))
    .expect_err("revoked issuer cannot authorize a child");
    assert!(matches!(
        error,
        crate::context::process_invocation::ProcessInvocationError::DirectoryCapture(_)
    ));
    assert!(!marker.exists());
}

#[cfg(unix)]
#[tokio::test]
async fn pending_process_stops_when_selected_directory_grant_revokes_after_provider_swap() {
    let _serial = runmat_filesystem::provider_override_lock();
    let selected = Arc::new(RevocableDirectory::new());
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            crate::context::RuntimeServicePorts::default().with_process(Rc::new(DirectProcess)),
        );
    let directory = tempfile::tempdir().expect("marker directory");
    let marker = directory.path().join("child-pid");
    let escaped = marker.to_string_lossy().replace('\'', "'\\''");
    let command = format!("printf '%s' \"$$\" > '{escaped}'; exec /bin/sleep 30");
    let mut call = Box::pin(run(Some(&runtime), &command));
    let pid = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(text) = std::fs::read_to_string(&marker) {
                if let Ok(pid) = text.parse::<u32>() {
                    break pid;
                }
            }
            tokio::select! {
                result = &mut call => panic!("process ended before revocation: {result:?}"),
                () = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
        }
    })
    .await
    .expect("child must start");
    let _replacement = runmat_filesystem::replace_provider(Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir("/replacement"),
    ));
    selected.revoke();
    let error = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("revocation must wake pending process")
        .expect_err("revoked authority cannot publish command output");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:system:StaleFilesystem")
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while runmat_process_host::is_process_alive(pid) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("revoked selected grant must terminate the owned child");
}
