//! The fixture owns its namespace until the detached authority acknowledges stop.
//! This is test cleanup, never authority for production task cancellation.
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(super) struct SupervisorFixture {
    namespace: Option<tempfile::TempDir>,
    root: PathBuf,
}

impl SupervisorFixture {
    pub(super) fn new(name: &str) -> Self {
        let namespace = tempfile::tempdir().unwrap();
        let root = namespace.path().join(name);
        Self {
            namespace: Some(namespace),
            root,
        }
    }

    pub(super) fn root(&self) -> &Path {
        &self.root
    }

    pub(super) fn request_stop(&self) -> std::io::Result<()> {
        std::fs::write(self.root.join("test-stop"), b"stop")
    }

    fn released(&self) -> bool {
        self.root.join("test-stopped").is_file()
            && runmat_filesystem::ExclusiveFileLock::try_acquire(&self.root.join("supervisor.lock"))
                .is_ok_and(|owner| owner.is_some())
    }
}

impl Drop for SupervisorFixture {
    fn drop(&mut self) {
        let requested = self.request_stop();
        let started = Instant::now();
        if requested.is_ok() {
            while started.elapsed() < Duration::from_secs(5) {
                if self.released() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        // Never remove an uncertain live provider's namespace during unwinding.
        // Preserve its stop request even if bootstrap completes after our timeout.
        if let Some(namespace) = self.namespace.take() {
            let retained = namespace.keep();
            eprintln!(
                "supervisor fixture shutdown unconfirmed; retained {} ({requested:?})",
                retained.display()
            );
        }
    }
}

pub(super) async fn verify_unwind() {
    let fixture = SupervisorFixture::new("unwind-authority");
    let root = fixture.root().to_path_buf();
    let namespace = root.parent().unwrap().to_path_buf();
    let execution = super::NativeExecutionConfig::for_current_executable().unwrap();
    let config = super::LocalSupervisorConfig::new(
        execution,
        super::SupervisorPaths::new(root.clone()).unwrap(),
    )
    .unwrap();
    let client = super::LocalSupervisorClient::new(config);
    client.connect_or_start().await.unwrap();
    client.ping().await.unwrap();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _fixture = fixture;
        panic!("intentional fixture unwind");
    }));
    assert!(unwound.is_err());
    assert!(
        !namespace.exists(),
        "acknowledged fixture namespace was not collected"
    );
    println!("actual detached supervisor fixture unwind cleanup passed");
}
