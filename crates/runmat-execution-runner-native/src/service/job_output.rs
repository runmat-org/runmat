//! One caller cursor across durable attempts; no runtime or result payload is retained.
use runmat_runtime::context::{ProcessConsoleDecoder, RuntimeContext};
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::SharedJobHandle;

use crate::durable::DurableJobBridge;
use crate::supervisor::{RawProcessOutputAttempt, RawProcessOutputCursor};
use crate::RawOutputState;

pub(super) fn wait<'a>(
    record: std::rc::Rc<super::DurableJobReceipt>,
    bridge: &'a DurableJobBridge,
    handle: SharedJobHandle,
    runtime: &'a RuntimeContext,
) -> Result<
    runmat_runtime::execution::AdmittedExecutionSupervisor<'a, (), ExecutionServiceError>,
    ExecutionServiceError,
> {
    record.authority.check_context(runtime)?;
    runmat_runtime::execution::admit_execution_supervisor_with(runtime, move || async move {
        loop {
            record.authority.check_context(runtime)?;
            let progress = match record.output.try_borrow_mut() {
                Ok(mut output) => output.pump(bridge, &handle, runtime),
                Err(_) => {
                    return Err(ExecutionServiceError::Infrastructure(
                        "application output delivery is already in progress".into(),
                    ))
                }
            };
            let progress = match progress {
                Ok(progress) => progress,
                Err(_) => {
                    return Err(record.output.borrow_mut().settle_failure(
                        bridge,
                        &handle,
                        runtime,
                        &record.authority,
                    ))
                }
            };
            if record
                .output
                .try_borrow()
                .is_ok_and(|output| output.complete())
            {
                return Ok(());
            }
            if progress {
                tokio::task::yield_now().await;
            } else {
                tokio::time::sleep(crate::durable::AWAIT_POLL_INTERVAL).await;
            }
        }
    })
}

pub(super) struct DurableCallerOutput {
    ordinal: u32,
    active: Option<RawProcessOutputAttempt>,
    cursor: RawProcessOutputCursor,
    decoders: [ProcessConsoleDecoder; 2],
    eof: [bool; 2],
    complete: bool,
    failure: Option<ExecutionServiceError>,
    failure_settled: bool,
}

impl Default for DurableCallerOutput {
    fn default() -> Self {
        Self {
            ordinal: 1,
            active: None,
            cursor: Default::default(),
            decoders: Default::default(),
            eof: [false; 2],
            complete: false,
            failure: None,
            failure_settled: false,
        }
    }
}

impl DurableCallerOutput {
    pub(super) fn complete(&self) -> bool {
        self.complete
    }

    pub(super) fn settle_failure(
        &mut self,
        bridge: &DurableJobBridge,
        handle: &SharedJobHandle,
        runtime: &RuntimeContext,
        authority: &super::job_receipt::JobResultAuthority,
    ) -> ExecutionServiceError {
        let output_error = self
            .failure
            .as_ref()
            .expect("failed durable delivery retained")
            .clone();
        if self.failure_settled {
            return output_error;
        }
        use runmat_execution_artifact::ProgramExecutionResponse;
        let primary = match bridge.try_response(handle) {
            Ok(Some(ProgramExecutionResponse::RuntimeFailure { failure })) => Some(
                authority
                    .decode_failure(runtime, &failure)
                    .unwrap_or_else(|error| error.into_runtime_error_in_context(Some(runtime))),
            ),
            Ok(Some(ProgramExecutionResponse::Failure { message })) => Some(
                ExecutionServiceError::Failed(message).into_runtime_error_in_context(Some(runtime)),
            ),
            Err(error) => Some(error.into_runtime_error_in_context(Some(runtime))),
            _ => None,
        };
        let error = match primary {
            Some(primary) => runmat_runtime::runtime_error::complete_with_cleanup::<()>(
                Err(primary),
                Err(output_error.into_runtime_error_in_context(Some(runtime))),
            )
            .expect_err("original failure cannot become success")
            .into(),
            None => output_error,
        };
        self.failure = Some(error.clone());
        self.failure_settled = true;
        error
    }

    pub(super) fn pump(
        &mut self,
        bridge: &DurableJobBridge,
        handle: &SharedJobHandle,
        runtime: &RuntimeContext,
    ) -> Result<bool, ExecutionServiceError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.pump_open(bridge, handle, runtime);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
        }
        result
    }

    fn pump_open(
        &mut self,
        bridge: &DurableJobBridge,
        handle: &SharedJobHandle,
        runtime: &RuntimeContext,
    ) -> Result<bool, ExecutionServiceError> {
        if self.complete {
            return Ok(false);
        }
        if self.active.is_none() {
            let lookup = bridge.raw_output_attempt(handle.clone(), self.ordinal)?;
            if lookup.task.is_none() {
                return Err(ExecutionServiceError::UnknownHandle);
            }
            self.active = lookup.attempt;
            if self.active.is_none() {
                self.complete = lookup.state.is_terminal();
                return Ok(self.complete);
            }
        }
        let active = self.active.expect("original durable attempt selected");
        let Some(page) = bridge.raw_output_page(handle.clone(), active.attempt_id, self.cursor)?
        else {
            let lookup = bridge.raw_output_attempt(handle.clone(), self.ordinal)?;
            let refreshed = lookup
                .attempt
                .filter(|attempt| attempt.attempt_id == active.attempt_id)
                .ok_or_else(|| invalid("durable output attempt changed identity"))?;
            self.active = Some(refreshed);
            if refreshed.not_started {
                self.advance();
                return Ok(true);
            }
            return if refreshed.terminal.is_some() {
                Err(invalid(
                    "closed durable attempt has no original output journal",
                ))
            } else {
                Ok(false)
            };
        };
        let mut progress = page.next != self.cursor;
        for (index, bytes, state, stream) in [
            (
                0,
                page.stdout.as_slice(),
                page.stdout_state,
                runmat_runtime::console::ConsoleStream::Stdout,
            ),
            (
                1,
                page.stderr.as_slice(),
                page.stderr_state,
                runmat_runtime::console::ConsoleStream::Stderr,
            ),
        ] {
            if self.eof[index] {
                if !bytes.is_empty() {
                    return Err(invalid("durable stream appended bytes after EOF"));
                }
                continue;
            }
            let eof = state != RawOutputState::Reading;
            if !bytes.is_empty() || eof {
                self.decoders[index].publish(stream, bytes, eof, runtime)?;
                progress = true;
            }
            if matches!(
                state,
                RawOutputState::ReadFailed | RawOutputState::CaptureFailed
            ) {
                return Err(invalid(
                    "durable application output was not completely retained",
                ));
            }
            self.eof[index] = eof;
        }
        self.cursor = page.next;
        if self.eof == [true; 2] && page.retired {
            self.advance();
            return Ok(true);
        }
        if !progress {
            let lookup = bridge.raw_output_attempt(handle.clone(), self.ordinal)?;
            if lookup.state.is_terminal() && !page.retired {
                // The page above was read before this terminal observation.
                // The producer marks its journal retired before it returns the
                // outcome whose report makes the job terminal, so only a page
                // read after the terminal state can prove a missing retirement.
                let fresh = bridge
                    .raw_output_page(handle.clone(), active.attempt_id, self.cursor)?
                    .ok_or_else(|| invalid("durable output journal disappeared"))?;
                if fresh.retired || fresh.next != self.cursor {
                    return Ok(true);
                }
                return Err(ExecutionServiceError::WorkerLost(
                    "durable output has no verified process retirement".into(),
                ));
            }
        }
        Ok(progress)
    }

    fn advance(&mut self) {
        // The extra ordinal represents exhaustion after the final u16 attempt.
        self.ordinal += 1;
        self.active = None;
        self.cursor = Default::default();
        self.decoders = Default::default();
        self.eof = [false; 2];
    }
}

fn invalid(message: &'static str) -> ExecutionServiceError {
    ExecutionServiceError::Infrastructure(message.into())
}
