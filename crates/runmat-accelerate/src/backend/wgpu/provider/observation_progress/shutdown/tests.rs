use super::*;
use super::super::{Command, State};
use std::sync::{atomic::AtomicBool, atomic::AtomicU32, mpsc::sync_channel};

#[test]
fn process_exit_drains_worker_after_main_thread_local_destruction() {
    const CHILD: &str = "RUNMAT_GPU_SHUTDOWN_EXIT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // Exercise Rust thread state as well as the native worker. The test
        // harness returns normally through Rust's runtime before C finalizers.
        let _ = std::thread::current();
        prepare().expect("register process shutdown");
        let (sender, receiver) = sync_channel(1);
        let control = Arc::new(Control {
            sender,
            state: Arc::new(State {
                used: AtomicU32::new(0),
                limit: 1,
                stopped: Arc::new(AtomicBool::new(false)),
            }),
        });
        spawn(&control, move || {
            assert!(matches!(receiver.recv(), Ok(Command::Shutdown)));
        })
        .expect("start shutdown worker");
        // Keep the sender alive until the process callback upgrades its weak
        // owner; otherwise this would only test ordinary channel closure.
        // A process-lifetime owner models the embedding host's live device.
        static OWNER: OnceLock<Arc<Control>> = OnceLock::new();
        assert!(OWNER.set(control).is_ok());
        return;
    }

    let module = module_path!().split_once("::").unwrap().1;
    let name = format!("{module}::process_exit_drains_worker_after_main_thread_local_destruction");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture"])
        .env(CHILD, "1")
        .output()
        .expect("execute shutdown child");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "child exit: {stderr}");
    assert!(!stderr.contains("panicked"), "shutdown panic: {stderr}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}
