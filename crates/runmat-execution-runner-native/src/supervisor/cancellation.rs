//! Durable controls validate original handle authority under the same lock as mutation.
use super::filesystem::unix_millis;
use super::{LocalJobRecord, LocalJobState, LocalSupervisor};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{CancellationReason, JobHandle, JobId};
use runmat_process_host::terminate_process_tree;

impl LocalSupervisor {
    pub async fn cancel(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        self.cancel_original(job_id, None, CancellationReason::User)
            .await
    }

    pub async fn cancel_handle(
        &self,
        handle: JobHandle,
        reason: CancellationReason,
    ) -> NativeExecutionResult<LocalJobRecord> {
        self.cancel_original(handle.id, Some(handle), reason).await
    }

    async fn cancel_original(
        &self,
        job_id: JobId,
        expected: Option<JobHandle>,
        reason: CancellationReason,
    ) -> NativeExecutionResult<LocalJobRecord> {
        self.reconcile().await?;
        let mut store = self.store.lock().await;
        let mut record = store.record(job_id)?;
        if expected
            .as_ref()
            .is_some_and(|handle| *handle != record.handle)
        {
            return Err(NativeExecutionError::Protocol(
                "durable cancellation handle differs from its original admission".into(),
            ));
        }
        if store.retained_submission(job_id)?.is_some() {
            return self.cancel_program(&mut store, record, reason).await;
        }
        // The legacy script path has no admitted task/cause authority. Do not
        // pretend it implements the generic runtime's governed cancellation.
        if expected.is_some() {
            return Err(NativeExecutionError::Protocol(
                "durable runtime cancellation requires original program admission".into(),
            ));
        }
        if record.state.is_terminal() {
            return Ok(record);
        }
        let process_id = record.driver_process_id.or_else(|| {
            store
                .driver_marker(job_id)
                .ok()
                .flatten()
                .map(|value| value.process_id)
        });
        let now = unix_millis();
        if process_id.is_none() {
            record.state = LocalJobState::Cancelled;
            record.updated_unix_millis = now;
            record.message = Some("cancelled before driver start".into());
            store.write_record(&mut record)?;
            return Ok(record);
        }
        record.state = LocalJobState::Cancelling;
        record.driver_process_id = process_id;
        record.updated_unix_millis = now;
        store.write_record(&mut record)?;
        terminate_process_tree(process_id.expect("checked above"))
            .await
            .map_err(protocol_io)?;
        record.state = LocalJobState::Cancelled;
        record.driver_process_id = None;
        record.updated_unix_millis = unix_millis();
        record.message = Some("cancelled by user".into());
        store.write_record(&mut record)?;
        Ok(record)
    }
}

fn protocol_io(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
