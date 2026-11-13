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
}

// The kernel lease can be released before run_child writes its acknowledgement.
// Both are required before collecting the namespace or starting a replacement.
pub(super) fn supervisor_stopped(root: &Path) -> bool {
    root.join("test-stopped").is_file()
        && runmat_filesystem::ExclusiveFileLock::try_acquire(&root.join("supervisor.lock"))
            .is_ok_and(|owner| owner.is_some())
}

fn verify_stop_ack_boundary() {
    let fixture = SupervisorFixture::new("stop-ack-boundary");
    std::fs::create_dir_all(fixture.root()).unwrap();
    assert!(
        !supervisor_stopped(fixture.root()),
        "a released kernel lease does not acknowledge child teardown"
    );
    let authority =
        runmat_filesystem::ExclusiveFileLock::try_acquire(&fixture.root().join("supervisor.lock"))
            .unwrap()
            .unwrap();
    std::fs::write(fixture.root().join("test-stopped"), b"stopped").unwrap();
    assert!(
        !supervisor_stopped(fixture.root()),
        "the acknowledgement does not release a live kernel authority"
    );
    drop(authority);
    assert!(supervisor_stopped(fixture.root()));
    println!("supervisor stop requires child acknowledgement and kernel release");
}

impl Drop for SupervisorFixture {
    fn drop(&mut self) {
        let requested = self.request_stop();
        let started = Instant::now();
        if requested.is_ok() {
            while started.elapsed() < Duration::from_secs(5) {
                if supervisor_stopped(&self.root) {
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
    verify_stop_ack_boundary();
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
