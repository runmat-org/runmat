//! Exact job/attempt reads without cloning a full execution history or payload.
use runmat_execution::{identity::AttemptId, process_output::ProcessOutputBytes, JobHandle};

use super::{
    LocalSupervisor, RawProcessOutputAttempt, RawProcessOutputCursor, RawProcessOutputLookup,
    RawProcessOutputPage,
};
use crate::process_output::{
    RawOutputBudget, RawOutputIdentity, RawOutputReader, RawOutputStream, OUTPUT_PAGE_BYTES,
};
use crate::{NativeExecutionError, NativeExecutionResult};

impl LocalSupervisor {
    pub async fn raw_output_attempt(
        &self,
        handle: JobHandle,
        ordinal: u32,
    ) -> NativeExecutionResult<RawProcessOutputLookup> {
        if ordinal == 0 || ordinal > u32::from(u16::MAX) + 1 {
            return Err(invalid("raw output ordinal exceeds the attempt inventory"));
        }
        let store = self.store.lock().await;
        let record = store.record(handle.id)?;
        if record.handle != handle {
            return Err(invalid("raw output handle differs from original job"));
        }
        let scheduler = self.scheduler.lock().await;
        let task = scheduler.task_for_job(handle.id);
        let attempt = task.and_then(|task| {
            scheduler
                .driver
                .snapshot_ref()
                .attempts
                .values()
                .find(|attempt| {
                    attempt.request.task_id == task && u32::from(attempt.request.ordinal) == ordinal
                })
                .map(|attempt| RawProcessOutputAttempt {
                    attempt_id: attempt.request.id,
                    ordinal: attempt.request.ordinal,
                    terminal: attempt
                        .terminal
                        .as_ref()
                        .map(|terminal| terminal.disposition),
                    not_started: attempt.certifies_not_started(),
                })
        });
        if let Some(task) = task {
            let original = scheduler
                .driver
                .snapshot_ref()
                .tasks
                .get(&task)
                .ok_or_else(|| invalid("raw output task is absent"))?;
            if attempt.is_none() && ordinal <= u32::from(original.attempt_count) {
                return Err(invalid("raw output attempt inventory is incomplete"));
            }
        }
        Ok(RawProcessOutputLookup {
            handle,
            task,
            ordinal,
            state: record.state,
            workload_kind: record.workload_kind,
            retain_until_unix_millis: record.retain_until_unix_millis,
            attempt,
        })
    }

    pub async fn raw_output_page(
        &self,
        handle: JobHandle,
        attempt_id: AttemptId,
        cursor: RawProcessOutputCursor,
    ) -> NativeExecutionResult<Option<RawProcessOutputPage>> {
        let (root, identity, maximum) = {
            let store = self.store.lock().await;
            if store.record(handle.id)?.handle != handle {
                return Err(invalid("raw output handle differs from original job"));
            }
            let scheduler = self.scheduler.lock().await;
            let attempt = scheduler
                .driver
                .snapshot_ref()
                .attempts
                .get(&attempt_id)
                .ok_or_else(|| invalid("raw output attempt is absent"))?;
            if scheduler.route(attempt.request.task_id)? != handle.id {
                return Err(invalid("raw output attempt differs from original job"));
            }
            let identity = RawOutputIdentity::for_attempt(&attempt.request)?;
            // Read-only upper bound from the original grant. The writer's
            // constructor separately subtracts authenticated materialization;
            // this reader cannot issue or enlarge a writing reservation.
            let maximum = RawOutputBudget::from_grant(
                attempt.request.task.resources.scratch_bytes,
                0,
                attempt.request.task.console,
            )?;
            (
                store
                    .job_dir(handle.id)
                    .join(format!("{attempt_id}.output")),
                identity,
                maximum,
            )
        };
        // Read no pages under scheduler/store authority. The immutable header
        // is published last, so no header means this attempt has no prefix yet.
        match std::fs::metadata(root.join("original.json")) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(crate::RawOutputError::Io(error).into()),
            Ok(_) => {}
        }
        let reader = RawOutputReader::open(root, identity, maximum)?;
        let mut buffer = [0; OUTPUT_PAGE_BYTES];
        let (stdout_count, stdout_state, stdout_retired) =
            reader.read_into(RawOutputStream::Stdout, cursor.stdout, &mut buffer)?;
        let stdout = ProcessOutputBytes::new(buffer[..stdout_count].to_vec());
        let (stderr_count, stderr_state, stderr_retired) =
            reader.read_into(RawOutputStream::Stderr, cursor.stderr, &mut buffer)?;
        let stderr = ProcessOutputBytes::new(buffer[..stderr_count].to_vec());
        Ok(Some(RawProcessOutputPage {
            attempt_id,
            stdout,
            stderr,
            next: RawProcessOutputCursor {
                stdout: cursor
                    .stdout
                    .checked_add(stdout_count as u64)
                    .ok_or_else(|| invalid("raw stdout cursor overflows"))?,
                stderr: cursor
                    .stderr
                    .checked_add(stderr_count as u64)
                    .ok_or_else(|| invalid("raw stderr cursor overflows"))?,
            },
            stdout_state,
            stderr_state,
            retired: stdout_retired && stderr_retired,
        }))
    }
}

fn invalid(message: &'static str) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}
