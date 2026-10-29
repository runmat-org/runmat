//! One exact in-flight frame read across invocation cancellation checkpoints.
use std::time::Duration;

use runmat_process_host::ipc::FrameLimits;
use tokio::io::AsyncRead;

use super::{read_message, PythonHostMessage, PythonWireError};
use crate::context::RuntimeContext;

pub(super) enum InvocationRead {
    Message(PythonHostMessage),
    Cancelled,
}

/// One owner retains partial header/payload progress across cancellation polls.
/// Uses the shared frame codec unchanged, like the native-FFI invocation reader.
pub(super) async fn read_invocation_message(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
    runtime: &RuntimeContext,
) -> Result<InvocationRead, PythonWireError> {
    let read = read_message(reader, limits);
    tokio::pin!(read);
    loop {
        if runtime.is_cancelled() {
            return Ok(InvocationRead::Cancelled);
        }
        tokio::select! {
            message = &mut read => return message.map(InvocationRead::Message),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
}

#[cfg(test)]
#[path = "invocation_read/tests.rs"]
mod tests;
