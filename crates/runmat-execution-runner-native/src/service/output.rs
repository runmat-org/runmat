//! Caller-local cursors and text projection; no context is retained by a worker.
use std::sync::Arc;

use runmat_runtime::context::{ProcessConsoleDecoder, RuntimeContext};
use runmat_runtime::execution::ExecutionServiceError;

use crate::driver::TaskCompletion;
use crate::process_output::{
    RawOutputError, RawOutputReadHandle, RawOutputState, RawOutputStream, OUTPUT_PAGE_BYTES,
};

#[derive(Default)]
pub(super) struct CallerOutput {
    next_attempt: usize,
    active: Option<AttemptOutput>,
    failure: Option<ExecutionServiceError>,
    failure_settled: bool,
}

struct AttemptOutput {
    reader: RawOutputReadHandle,
    offsets: [u64; 2],
    decoders: [ProcessConsoleDecoder; 2],
    finished: [bool; 2],
}

impl CallerOutput {
    pub(super) fn pending(&self, completion: &TaskCompletion) -> bool {
        self.failure.is_some()
            || self.active.is_some()
            || completion.raw_output_reader(self.next_attempt).is_some()
    }

    pub(super) fn pump(
        &mut self,
        completion: &TaskCompletion,
        runtime: &RuntimeContext,
    ) -> Result<bool, ExecutionServiceError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.pump_open(completion, runtime);
        if let Err(error) = &result {
            // Text emission may fail after publishing part of a page. Never
            // replay that page on a repeated fetch and duplicate its effects.
            self.failure = Some(error.clone());
        }
        result
    }

    fn pump_open(
        &mut self,
        completion: &TaskCompletion,
        runtime: &RuntimeContext,
    ) -> Result<bool, ExecutionServiceError> {
        if self.active.is_none() {
            let Some(reader) = completion.raw_output_reader(self.next_attempt) else {
                return match completion.raw_output_failure() {
                    Some(error) => Err(output_error(error, runtime)),
                    None => Ok(false),
                };
            };
            self.next_attempt += 1;
            self.active = Some(AttemptOutput {
                reader,
                offsets: [0; 2],
                decoders: std::array::from_fn(|_| ProcessConsoleDecoder::default()),
                finished: [false; 2],
            });
        }
        let active = self
            .active
            .as_mut()
            .expect("original output reader selected");
        // Completion joins the pipe owners before publication. Snapshot it
        // before reading their heads so a later completion cannot invalidate
        // an earlier, still-Reading page.
        let terminal_before_page = completion.verified_terminal().is_some();
        let mut buffer = [0; OUTPUT_PAGE_BYTES];
        let mut progress = false;
        for (index, stream, console_stream) in [
            (
                0,
                RawOutputStream::Stdout,
                runmat_runtime::console::ConsoleStream::Stdout,
            ),
            (
                1,
                RawOutputStream::Stderr,
                runmat_runtime::console::ConsoleStream::Stderr,
            ),
        ] {
            if active.finished[index] {
                continue;
            }
            let (count, state, _) = active
                .reader
                .read_into(stream, active.offsets[index], &mut buffer)
                .map_err(|error| output_error(Arc::new(error), runtime))?;
            let eof = state != RawOutputState::Reading;
            if count != 0 || eof {
                active.decoders[index].publish(console_stream, &buffer[..count], eof, runtime)?;
                active.offsets[index] += count as u64;
                progress = true;
            }
            if matches!(
                state,
                RawOutputState::ReadFailed | RawOutputState::CaptureFailed
            ) {
                return Err(output_error(
                    completion.raw_output_failure().unwrap_or_else(|| {
                        Arc::new(RawOutputError::Invalid(
                            "application stream was not completely retained",
                        ))
                    }),
                    runtime,
                ));
            }
            active.finished[index] = eof;
        }
        if active.finished == [true; 2] {
            self.active = None;
            progress = true;
        }
        if !progress {
            if let Some(error) = completion.raw_output_failure() {
                return Err(output_error(error, runtime));
            }
            if self.active.is_some() && terminal_before_page {
                // Terminal completion follows the original pipe owners' join.
                // A remaining Reading head therefore cannot acquire more bytes;
                // refusing publication is preferable to an endless pending wait
                // or inventing a successful end-of-stream record.
                return Err(output_error(
                    Arc::new(RawOutputError::Invalid(
                        "completed attempt has an unfinished application stream",
                    )),
                    runtime,
                ));
            }
        }
        Ok(progress)
    }

    pub(super) fn settle_failure(
        &mut self,
        completion: &TaskCompletion,
        runtime: &RuntimeContext,
        graph: Option<&dyn runmat_runtime::execution::GraphResultReceipt>,
    ) -> ExecutionServiceError {
        let error = self
            .failure
            .as_ref()
            .expect("failed delivery retained")
            .clone();
        if self.failure_settled {
            return error;
        }
        let primary = match completion.original_failure() {
            Some(crate::driver::TransferFailure::Runtime(failure)) => Some(match graph {
                Some(receipt) => receipt
                    .decode_failure_in_context(runtime, &failure)
                    .map_err(|error| error.into_runtime_error_in_context(Some(runtime))),
                None => Ok(ExecutionServiceError::RuntimeFailure(failure)
                    .into_runtime_error_in_context(Some(runtime))),
            }),
            Some(crate::driver::TransferFailure::Execution(message)) => {
                Some(Ok(ExecutionServiceError::Failed(message)
                    .into_runtime_error_in_context(Some(runtime))))
            }
            Some(crate::driver::TransferFailure::RetirementUnverified(message)) => {
                Some(Ok(ExecutionServiceError::Infrastructure(message)
                    .into_runtime_error_in_context(Some(runtime))))
            }
            _ => None,
        };
        let error = match primary {
            Some(primary) => {
                let primary = match primary {
                    Ok(error) | Err(error) => error,
                };
                runmat_runtime::runtime_error::complete_with_cleanup::<()>(
                    Err(primary),
                    Err(error.into_runtime_error_in_context(Some(runtime))),
                )
                .expect_err("original failure cannot become success")
                .into()
            }
            None => error,
        };
        self.failure = Some(error.clone());
        self.failure_settled = true;
        error
    }
}

pub(super) fn output_error(
    error: Arc<RawOutputError>,
    runtime: &RuntimeContext,
) -> ExecutionServiceError {
    if let Some(cause) = error.primary_allocation_failure() {
        return runmat_runtime::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:parallel:ProcessOutput",
            "process output control allocation",
            cause,
        )
        .into();
    }
    runmat_runtime::build_runtime_error("execution worker application output failed")
        .with_identifier(format!(
            "{}:parallel:ProcessOutput",
            runtime.error_namespace()
        ))
        .with_source(error)
        .build()
        .into()
}
