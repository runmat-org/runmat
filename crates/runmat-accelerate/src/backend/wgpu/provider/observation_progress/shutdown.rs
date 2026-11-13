//! Retain native worker joins until they finish, and drain them before driver
//! process finalizers. Ordinary provider/cancellation Drop remains nonblocking.

use super::{Control, State};
#[path = "shutdown/native_thread.rs"]
mod native_thread;
use std::{
    io,
    sync::{Arc, Mutex, OnceLock, Weak},
    thread::JoinHandle,
};

struct Worker {
    control: Weak<Control>,
    thread: JoinHandle<RetiredWorker>,
}

// Transfer the final device owner to the joining thread. A device dropped by
// its worker during atexit can unload DXC, whose DLL finalizer waits for the
// C-runtime exit lock already held by the thread joining that worker.
// The thread result retains the same state allocation; it does not clone GPU
// storage or introduce another progress thread.
type RetiredWorker = (Arc<State>, std::thread::Result<()>);

#[derive(Default)]
struct Registry {
    closing: bool,
    workers: Vec<Worker>,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
static EXIT_HOOK: OnceLock<Result<(), i32>> = OnceLock::new();

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

/// Called after device initialization, so the later-registered process callback
/// runs before the driver's own C-runtime finalizers (reverse registration).
pub(super) fn prepare() -> io::Result<()> {
    registry();
    let installed = EXIT_HOOK.get_or_init(|| {
        // SAFETY: static C-ABI callback, no captured or borrowed data; the
        // callback contains unwinding and uses only process-lifetime state.
        let result = unsafe { libc::atexit(shutdown_at_exit) };
        if result == 0 {
            Ok(())
        } else {
            Err(result)
        }
    });
    installed.as_ref().map_err(|code| {
        io::Error::other(format!("GPU worker exit-hook registration failed: {code}"))
    })?;
    let registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    if registry.closing {
        return Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "GPU workers are shutting down",
        ));
    }
    Ok(())
}

pub(super) fn spawn(control: &Arc<Control>, run: impl FnOnce() + Send + 'static) -> io::Result<()> {
    let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
    // A finished thread has already retired its device access. Reap handles
    // during subsequent device creation instead of retaining historical owners.
    let mut finished = Vec::new();
    let mut index = 0;
    while index < registry.workers.len() {
        if registry.workers[index].thread.is_finished() {
            finished.push(registry.workers.swap_remove(index).thread);
        } else {
            index += 1;
        }
    }
    drop(registry);
    for thread in finished {
        let _ = thread.join();
    }
    let mut registry = self::registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if registry.closing {
        return Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "GPU workers are shutting down",
        ));
    }
    // Spawn and publish under one admission lock: an exit snapshot cannot miss
    // a started worker. The worker itself never needs this registry mutex.
    registry
        .workers
        .try_reserve(1)
        .map_err(|error| io::Error::other(format!("GPU worker registry capacity: {error}")))?;
    let thread = spawn_retained(Arc::clone(&control.state), run)?;
    registry.workers.push(Worker {
        control: Arc::downgrade(control),
        thread,
    });
    Ok(())
}

fn spawn_retained(
    retirement_owner: Arc<State>,
    run: impl FnOnce() + Send + 'static,
) -> io::Result<JoinHandle<RetiredWorker>> {
    std::thread::Builder::new()
        .name("runmat-gpu-observation".into())
        .spawn(move || {
            // Preserve the owner on both normal return and unwind. The panic
            // hook still reports a worker panic; joining never resumes it.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
            (retirement_owner, result)
        })
}

fn join_unless_current(thread: JoinHandle<RetiredWorker>) {
    // Internal completion callbacks do not exit the process. Nevertheless an
    // embedding host can do so: never deadlock by joining the calling worker.
    if !native_thread::is_current(&thread) {
        let _ = thread.join();
    }
}

extern "C" fn shutdown_at_exit() {
    let _ = std::panic::catch_unwind(shutdown);
}

fn shutdown() {
    let workers = {
        let mut registry = registry().lock().unwrap_or_else(|error| error.into_inner());
        registry.closing = true;
        std::mem::take(&mut registry.workers)
    };
    // Stop every queue before waiting on any worker. No join holds a registry
    // mutex or an emission permit, and callbacks can retire their owners.
    for worker in &workers {
        if let Some(control) = worker.control.upgrade() {
            control.stop();
        }
    }
    for worker in workers {
        join_unless_current(worker.thread);
    }
}

#[cfg(test)]
#[path = "shutdown/tests.rs"]
mod tests;
