use super::super::{Command, State};
use super::*;
use std::sync::{atomic::AtomicBool, atomic::AtomicU32, mpsc::sync_channel};

#[test]
fn final_device_owner_transfers_to_joiner_on_return_and_panic() {
    for panic in [false, true] {
        let state = Arc::new(State {
            used: AtomicU32::new(0),
            limit: 1,
            stopped: Arc::new(AtomicBool::new(false)),
            scoped_emission: Mutex::new(()),
            scoped_work: Mutex::new(Vec::new()),
            worker_closed: AtomicBool::new(false),
            retained_uncertain: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            access: None,
            health: None,
        });
        let weak = Arc::downgrade(&state);
        let thread = spawn_retained(state, move || {
            assert!(!panic, "worker unwind fixture");
        })
        .unwrap();
        let (owner, result) = thread.join().expect("retirement owner survives unwind");
        assert_eq!(result.is_err(), panic);
        assert!(weak.upgrade().is_some());
        drop(owner);
        assert!(weak.upgrade().is_none());
    }
}

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
                scoped_emission: Mutex::new(()),
                scoped_work: Mutex::new(Vec::new()),
                worker_closed: AtomicBool::new(false),
                retained_uncertain: AtomicBool::new(false),
                shutdown_requested: AtomicBool::new(false),
                access: None,
                health: None,
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
