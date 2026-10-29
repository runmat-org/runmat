use super::RunState;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

#[derive(Default)]
pub(super) struct Heartbeat {
    signal: Arc<(Mutex<bool>, Condvar)>,
}

impl Heartbeat {
    pub(super) fn start(&self, state: Weak<RunState>) -> std::io::Result<()> {
        let signal = Arc::clone(&self.signal);
        std::thread::Builder::new()
            .name("runmat-fea-usage".into())
            .spawn(move || loop {
                let (lock, wake) = &*signal;
                let stopped = lock.lock().unwrap_or_else(|e| e.into_inner());
                let (stopped, _) = wake
                    .wait_timeout_while(stopped, Duration::from_secs(30), |stopped| !*stopped)
                    .unwrap_or_else(|e| e.into_inner());
                if *stopped {
                    break;
                }
                drop(stopped);
                let Some(state) = state.upgrade() else {
                    break;
                };
                state.report();
            })?;
        Ok(())
    }

    pub(super) fn stop(&self) {
        let (lock, wake) = &*self.signal;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        wake.notify_all();
    }
}
