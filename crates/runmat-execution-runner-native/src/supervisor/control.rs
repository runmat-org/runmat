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
    reason: Mutex<Option<runmat_execution::CancellationReason>>,
    progress: Mutex<Option<crate::protocol::ProgramProgress>>,
    console: Option<Mutex<super::console_journal::ConsoleJournal>>,
}
impl DurableControl {
    pub(super) fn for_attempt(
        attempt: &super::DurableProgramAttempt,
        lease: &super::DurableAttemptLease,
    ) -> NativeExecutionResult<Self> {
        Ok(Self {
            console: super::console_journal::ConsoleJournal::create(attempt, lease)?
                .map(Mutex::new),
            ..Default::default()
        })
    }
    pub(super) fn stop(
        &self,
        delivered: NativeExecutionResult<runmat_execution::CancellationReason>,
    ) {
        match delivered {
            Ok(reason) => *self.reason.lock().expect("durable control poisoned") = Some(reason),
            Err(error) => {
                *self.failure.lock().expect("durable control poisoned") = Some(error.to_string())
            }
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
    fn cancellation_reason(&self) -> Option<runmat_execution::CancellationReason> {
        *self.reason.lock().expect("durable control poisoned")
    }
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
    fn record_console(
        &self,
        record: &runmat_execution::console::ConsoleRecord,
    ) -> Result<(), String> {
        self.console
            .as_ref()
            .ok_or("durable console publisher is absent")?
            .lock()
            .map_err(|_| "durable console publisher poisoned")?
            .push(record)
            .map_err(|error| error.to_string())
    }
    fn finish_console(
        &self,
        receipt: &runmat_execution::console::ConsoleReceipt,
    ) -> Result<(), String> {
        self.console
            .as_ref()
            .ok_or("durable console publisher is absent")?
            .lock()
            .map_err(|_| "durable console publisher poisoned")?
            .finish(receipt)
            .map_err(|error| error.to_string())
    }
}

pub(super) async fn watch(
    path: &Path,
    original: &AttemptRequest,
) -> NativeExecutionResult<runmat_execution::CancellationReason> {
    loop {
        if let Some(marker) = super::filesystem::read_optional_json::<
            crate::process_provider::cancellation::CancellationCommand,
        >(path)?
        {
            if marker.request != *original {
                return Err(NativeExecutionError::Protocol(
                    "durable control differs from the full original assignment".into(),
                ));
            }
            return crate::process_provider::cancellation::reason(original, marker.cause)
                .map_err(NativeExecutionError::Protocol);
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[cfg(all(test, unix))]
mod tests;
