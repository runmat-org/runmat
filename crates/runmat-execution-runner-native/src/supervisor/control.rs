//! Original fenced control delivery; transport faults are never user cancellation.
use crate::process_provider::ProcessControl;
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution_runner::AttemptRequest;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
pub(super) struct DurableControl {
    cancelled: AtomicBool,
    failure: Mutex<Option<String>>,
    reason: Mutex<Option<runmat_execution::CancellationReason>>,
    progress: Mutex<Option<crate::protocol::ProgramProgress>>,
    console: Option<Mutex<super::console_journal::ConsoleJournal>>,
    raw_identity: Option<crate::process_output::RawOutputIdentity>,
    raw_root: Option<PathBuf>,
    raw_started: AtomicBool,
    raw_failure: Mutex<Option<Arc<crate::RawOutputError>>>,
}
impl DurableControl {
    pub(super) fn for_attempt(
        attempt: &super::DurableProgramAttempt,
        lease: &super::DurableAttemptLease,
    ) -> NativeExecutionResult<Self> {
        Ok(Self {
            console: super::console_journal::ConsoleJournal::create(attempt, lease)?
                .map(Mutex::new),
            raw_identity: Some(crate::process_output::RawOutputIdentity::for_attempt(
                &attempt.request,
            )?),
            raw_root: Some(
                attempt
                    .job_directory
                    .join(format!("{}.output", attempt.request.id)),
            ),
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
impl crate::process_output::OutputDrainControl for DurableControl {
    fn raw_output_changed(&self) {
        // Detached callers discover durable committed pages through the
        // authenticated supervisor API, not an in-process caller mailbox.
    }
    fn stop_for_raw_output(&self, error: Arc<crate::RawOutputError>) {
        self.raw_failure
            .lock()
            .expect("durable raw output poisoned")
            .get_or_insert(error);
    }
    fn raw_output_failure(&self) -> Option<Arc<crate::RawOutputError>> {
        self.raw_failure
            .lock()
            .expect("durable raw output poisoned")
            .clone()
    }
}
impl ProcessControl for DurableControl {
    fn begin_raw_output(
        &self,
        attempt: &AttemptRequest,
        materialized: u64,
    ) -> Result<Mutex<crate::process_output::RawOutputJournal>, Arc<crate::RawOutputError>> {
        let identity =
            crate::process_output::RawOutputIdentity::for_attempt(attempt).map_err(Arc::new)?;
        if self.raw_identity.as_ref() != Some(&identity) {
            return Err(Arc::new(crate::RawOutputError::Invalid(
                "output writer differs from the leased original attempt",
            )));
        }
        if self
            .raw_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Arc::new(crate::RawOutputError::Invalid(
                "original attempt already started its output writer",
            )));
        }
        let root = self.raw_root.as_ref().ok_or_else(|| {
            Arc::new(crate::RawOutputError::Invalid(
                "output writer has no leased directory",
            ))
        })?;
        let journal = crate::process_output::RawOutputJournal::for_attempt(
            root.clone(),
            attempt,
            materialized,
        )
        .map_err(Arc::new)?;
        Ok(Mutex::new(journal))
    }
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
