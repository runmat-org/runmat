//! Original fenced control delivery; transport faults are never user cancellation.
use crate::process_provider::ProcessControl;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution_runner::AttemptRequest;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

#[derive(Default)]
pub(super) struct DurableControl {
    cancelled: AtomicBool,
    failure: Mutex<Option<String>>,
    progress: Mutex<Option<crate::protocol::ProgramProgress>>,
}
impl DurableControl {
    pub(super) fn stop(&self, delivered: NativeExecutionResult<()>) {
        if let Err(error) = delivered {
            *self.failure.lock().expect("durable control poisoned") = Some(error.to_string());
        }
        self.cancelled.store(true, Ordering::Release);
    }
    pub(super) fn latest_progress(&self) -> Option<crate::protocol::ProgramProgress> {
        self.progress
            .lock()
            .expect("durable progress poisoned")
            .clone()
    }
}
impl ProcessControl for DurableControl {
    fn cancellation(&self) -> &AtomicBool {
        &self.cancelled
    }
    fn failure(&self) -> Option<String> {
        self.failure
            .lock()
            .expect("durable control poisoned")
            .clone()
    }
    fn record_progress(&self, progress: crate::protocol::ProgramProgress) {
        *self.progress.lock().expect("durable progress poisoned") = Some(progress);
    }
}

pub(super) async fn watch(path: &Path, original: &AttemptRequest) -> NativeExecutionResult<()> {
    loop {
        if let Some(marker) = super::filesystem::read_optional_json::<AttemptRequest>(path)? {
            if marker != *original {
                return Err(NativeExecutionError::Protocol(
                    "durable control differs from the full original assignment".into(),
                ));
            }
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[cfg(all(test, unix))]
mod tests;
