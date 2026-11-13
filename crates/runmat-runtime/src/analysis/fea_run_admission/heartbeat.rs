use super::RunState;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

#[derive(Default)]
pub(super) struct Heartbeat {
    signal: Arc<(Mutex<bool>, Condvar)>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}
impl Heartbeat {
    pub(super) fn start(&self, state: Weak<RunState>, reports: bool) -> std::io::Result<()> {
        // Deadline enforcement never waits behind a network call or permit lock.
        let signal = Arc::clone(&self.signal);
        let monitor = state.clone();
        let deadline = std::thread::Builder::new()
            .name("runmat-fea-deadline".into())
            .stack_size(super::super::FEA_DEADLINE_MONITOR_STACK_BYTES)
            .spawn(move || loop {
                let Some(state) = monitor.upgrade() else {
                    break;
                };
                if state.check().is_err() {
                    break;
                }
                let duration = state.deadline_wait();
                drop(state);
                if wait(&signal, duration) {
                    break;
                }
            })?;
        self.threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(deadline);
        if reports {
            let signal = Arc::clone(&self.signal);
            let report = std::thread::Builder::new()
                .name("runmat-fea-usage".into())
                .stack_size(super::super::FEA_HEARTBEAT_STACK_BYTES)
                .spawn(move || loop {
                    let Some(current) = state.upgrade() else {
                        break;
                    };
                    let interval = current.heartbeat_millis();
                    drop(current);
                    if wait(&signal, Duration::from_millis(interval)) {
                        break;
                    }
                    let Some(current) = state.upgrade() else {
                        break;
                    };
                    if current.check().is_err() {
                        break;
                    }
                    current.report();
                })?;
            self.threads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(report);
        }
        Ok(())
    }
    pub(super) fn stop(&self) {
        let (lock, wake) = &*self.signal;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        wake.notify_all();
        // Join outside the handle lock; the final state can drop on a monitor
        // thread. Never attempt to join the current thread.
        let handles = std::mem::take(&mut *self.threads.lock().unwrap_or_else(|e| e.into_inner()));
        for handle in handles {
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }
}
fn wait(signal: &(Mutex<bool>, Condvar), duration: Duration) -> bool {
    let (lock, wake) = signal;
    let stopped = lock.lock().unwrap_or_else(|e| e.into_inner());
    let (stopped, _) = wake
        .wait_timeout_while(stopped, duration, |stopped| !*stopped)
        .unwrap_or_else(|e| e.into_inner());
    *stopped
}
