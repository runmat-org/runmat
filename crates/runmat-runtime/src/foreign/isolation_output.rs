//! Application pipes are drained on the caller's executor and console owner.
//!
//! There is no message queue or second complete transcript. Reads use a
//! poll-local buffer; only an incomplete UTF-8 suffix survives a poll. The
//! existing bounded stderr diagnostic prefix keeps its single producer.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use runmat_execution::resource::ExactShared;
use runmat_process_host::ChildOutput;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use tokio::io::{AsyncRead, ReadBuf};

use crate::console::ConsoleStream;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{RuntimeConsoleScope, RuntimeContext};
use crate::RuntimeError;

#[derive(Clone, Default)]
pub(super) struct HostOutput {
    owner: Option<ExactShared<OutputOwner>>,
}

struct OutputOwner {
    streams: RefCell<ApplicationOutput>,
    origin: RefCell<Option<RuntimeConsoleScope>>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl HostOutput {
    pub(super) fn attached(
        pipes: ChildOutput,
        runtime: Option<&RuntimeContext>,
    ) -> Result<Self, RuntimeError> {
        let error = |source| {
            host_allocation_runtime_error(
                runtime,
                "RunMat:Foreign:ConsoleAllocation",
                "foreign output owner allocation failed",
                source,
            )
        };
        let bytes = ExactShared::<OutputOwner>::requested_bytes();
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| error(HostAllocationError::Overflow))?,
            1,
        )
        .map_err(error)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(error)?,
        )
        .map_err(error)?;
        let claim = match (receipt.as_ref(), runtime) {
            (Some(receipt), Some(runtime)) => Some(
                receipt
                    .output_permit(runtime)
                    .map_err(error)?
                    .reserve_exact_claim(bytes)
                    .map_err(|source| error(HostAllocationError::Numeric(source)))?,
            ),
            (None, _) => None,
            (Some(_), None) => return Err(error(HostAllocationError::MissingManagedOwner)),
        };
        let streams = ApplicationOutput {
            stdout: Some(pipes.stdout),
            stderr: Some(pipes.stderr),
            stdout_tail: Utf8Tail::default(),
            stderr_tail: Utf8Tail::default(),
            stderr_diagnostics: pipes.stderr_diagnostics,
            failed: false,
        };
        let owner = ExactShared::try_new(OutputOwner {
            streams: RefCell::new(streams),
            origin: RefCell::new(runtime.map(RuntimeContext::console_scope)),
            _claim: claim,
        })
        .map_err(|_| {
            error(HostAllocationError::Numeric(
                ExactHostNumericAllocationError::AllocationFailed,
            ))
        })?;
        if let (Some(receipt), Some(runtime)) = (receipt.as_ref(), runtime) {
            receipt.checkpoint(runtime).map_err(error)?;
        }
        Ok(Self { owner: Some(owner) })
    }

    pub(super) fn failed(&self) -> bool {
        self.owner
            .as_ref()
            .is_some_and(|owner| owner.get().streams.borrow().failed)
    }

    /// Select the console owner at a protocol phase boundary. Pending MEX
    /// recovery keeps its original owner before a later request is dispatched.
    pub(super) fn select_origin(&self, runtime: Option<&RuntimeContext>) {
        if let Some(owner) = &self.owner {
            *owner.get().origin.borrow_mut() = runtime.map(RuntimeContext::console_scope);
        }
    }

    pub(super) fn poll_available(&self, context: &mut Context<'_>) -> Result<bool, RuntimeError> {
        let Some(owner) = self.owner.as_ref() else {
            return Ok(false);
        };
        let mut output = owner.get().streams.borrow_mut();
        let origin = owner.get().origin.borrow();
        let result = output.poll_drain(context, origin.as_ref());
        if result.is_err() {
            output.failed = true;
        }
        result
    }
}

pub(super) enum Completion {
    AvailableOutput,
    EndOfStreams,
}

#[derive(Default)]
struct Utf8Tail {
    bytes: [u8; 3],
    len: usize,
}

struct ApplicationOutput {
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
    stdout_tail: Utf8Tail,
    stderr_tail: Utf8Tail,
    stderr_diagnostics: runmat_process_host::ProcessOutputCaptureWriter,
    failed: bool,
}

/// Also pumps while `work` is suspended inside a callback or waiting for child
/// exit. An unscoped caller explicitly selects ordinary unmanaged console
/// semantics; selected invocations pass their original RuntimeContext.
pub(super) async fn drive<T>(
    output: &HostOutput,
    runtime: Option<&RuntimeContext>,
    completion: Completion,
    work: impl Future<Output = Result<T, RuntimeError>>,
) -> Result<T, RuntimeError> {
    output.select_origin(runtime);
    drive_selected(output, completion, work).await
}

pub(super) async fn drive_selected<T>(
    output: &HostOutput,
    completion: Completion,
    work: impl Future<Output = Result<T, RuntimeError>>,
) -> Result<T, RuntimeError> {
    let Some(owner) = output.owner.as_ref() else {
        return work.await;
    };
    tokio::pin!(work);
    let mut completed = None;
    std::future::poll_fn(|context| {
        if completed.is_none() {
            if let Poll::Ready(result) = work.as_mut().poll(context) {
                completed = Some(result);
            }
        }
        let progress = match output.poll_available(context) {
            Ok(progress) => progress,
            Err(error) => {
                return Poll::Ready(match completed.take() {
                    Some(result) => crate::runtime_error::complete_with_cleanup(result, Err(error)),
                    None => Err(error),
                });
            }
        };
        if completed.is_some() {
            let drained = match completion {
                Completion::AvailableOutput => !progress,
                Completion::EndOfStreams => {
                    let output = owner.get().streams.borrow();
                    output.stdout.is_none() && output.stderr.is_none()
                }
            };
            if drained {
                return Poll::Ready(completed.take().expect("completed work"));
            }
        }
        if progress {
            context.waker().wake_by_ref();
        }
        Poll::Pending
    })
    .await
}

impl ApplicationOutput {
    fn poll_drain(
        &mut self,
        context: &mut Context<'_>,
        runtime: Option<&RuntimeConsoleScope>,
    ) -> Result<bool, RuntimeError> {
        // One chunk per stream per poll keeps a continuously printing child
        // from starving control replies, cancellation or the other stream.
        let stdout = poll_stream(
            &mut self.stdout,
            &mut self.stdout_tail,
            None,
            ConsoleStream::Stdout,
            context,
            runtime,
        )?;
        let stderr = poll_stream(
            &mut self.stderr,
            &mut self.stderr_tail,
            Some(&mut self.stderr_diagnostics),
            ConsoleStream::Stderr,
            context,
            runtime,
        )?;
        Ok(stdout || stderr)
    }
}

fn poll_stream<R: AsyncRead + Unpin>(
    reader: &mut Option<R>,
    tail: &mut Utf8Tail,
    mut diagnostics: Option<&mut runmat_process_host::ProcessOutputCaptureWriter>,
    stream: ConsoleStream,
    context: &mut Context<'_>,
    runtime: Option<&RuntimeConsoleScope>,
) -> Result<bool, RuntimeError> {
    let Some(pipe) = reader.as_mut() else {
        return Ok(false);
    };
    let mut buffer = [0u8; 8192 + 3];
    let prefix = tail.len;
    buffer[..prefix].copy_from_slice(&tail.bytes[..prefix]);
    let mut read = ReadBuf::new(&mut buffer[prefix..]);
    match Pin::new(pipe).poll_read(context, &mut read) {
        Poll::Pending => return Ok(false),
        Poll::Ready(Err(error)) => {
            if let Some(capture) = diagnostics.as_deref_mut() {
                capture.finish(Err(error.kind()));
            }
            return Err(
                crate::build_runtime_error("foreign application pipe read failed")
                    .with_identifier("RunMat:Foreign:ConsoleTransport")
                    .with_source(error)
                    .build(),
            );
        }
        Poll::Ready(Ok(())) => {}
    }
    let count = read.filled().len();
    if let Some(capture) = diagnostics.as_deref_mut() {
        capture.observe(read.filled()).map_err(|source| {
            crate::build_runtime_error("foreign stderr diagnostics could not be retained")
                .with_identifier("RunMat:Foreign:ConsoleTransport")
                .with_source(source)
                .build()
        })?;
        if count == 0 {
            capture.finish(Ok(()));
        }
    }
    let eof = count == 0;
    tail.len = 0;
    publish_utf8(
        &buffer[..prefix + count],
        eof,
        tail,
        |text| -> Result<(), RuntimeError> {
            match runtime {
                Some(scope) => scope.record(stream, text),
                None => crate::context::with_original_scope(None, || {
                    crate::console::record_console_output(stream, text)
                }),
            }
        },
    )?;
    if eof {
        *reader = None;
    }
    Ok(true)
}

fn publish_utf8<E>(
    mut bytes: &[u8],
    eof: bool,
    tail: &mut Utf8Tail,
    mut emit: impl FnMut(&str) -> Result<(), E>,
) -> Result<(), E> {
    while !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                emit(text)?;
                return Ok(());
            }
            Err(error) => {
                let (valid, rest) = bytes.split_at(error.valid_up_to());
                if !valid.is_empty() {
                    // The UTF-8 validator established this exact prefix.
                    emit(std::str::from_utf8(valid).expect("validated UTF-8 prefix"))?;
                }
                if let Some(invalid) = error.error_len() {
                    emit("\u{fffd}")?;
                    bytes = &rest[invalid..];
                } else {
                    if eof {
                        emit("\u{fffd}")?;
                    } else {
                        tail.bytes[..rest.len()].copy_from_slice(rest);
                        tail.len = rest.len();
                    }
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_utf8_and_invalid_bytes_follow_streaming_lossy_text_semantics() {
        for bytes in [
            b"before\xe2\x82\xac\xf0\x9f\x98\x80after".as_slice(),
            b"bad\xff\xe2\x82",
        ] {
            for split in 0..=bytes.len() {
                let mut tail = Utf8Tail::default();
                let mut text = String::new();
                publish_utf8(&bytes[..split], false, &mut tail, |part| {
                    text.push_str(part);
                    Ok::<_, ()>(())
                })
                .unwrap();
                let mut rest = tail.bytes[..tail.len].to_vec();
                rest.extend_from_slice(&bytes[split..]);
                tail.len = 0;
                publish_utf8(&rest, true, &mut tail, |part| {
                    text.push_str(part);
                    Ok::<_, ()>(())
                })
                .unwrap();
                assert_eq!(text, String::from_utf8_lossy(bytes), "split={split}");
                assert_eq!(tail.len, 0);
            }
        }
    }
}
