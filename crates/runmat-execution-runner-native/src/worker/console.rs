//! Source capture uses the existing runtime host; stdout stays a framed protocol.
use super::*;
use crate::protocol::WorkerProcessMessage;
use runmat_execution::console::{ConsoleReceipt, ConsoleRecord};
use runmat_runtime::context::PortableConsoleSink;
use std::rc::Rc;

/// Console records are framed onto stdout as they are produced, ahead of the
/// terminal message that carries their receipt.
struct FramedSink(FrameLimits);
impl PortableConsoleSink for FramedSink {
    fn emit(&self, record: &ConsoleRecord) -> Result<(), String> {
        let message = WorkerProcessMessage::Console {
            record: record.clone(),
        };
        let bytes = runmat_canonical_codec::encode_json_bounded(
            &message,
            u64::from(self.0.max_message_bytes),
        )
        .map_err(|error| error.to_string())?;
        runmat_process_host::ipc::write_payload_blocking(
            &mut std::io::stdout().lock(),
            &bytes,
            self.0,
        )
        .map_err(|error| error.to_string())
    }
}

pub(super) fn graph_host(
    controls: Option<runmat_execution::console::ConsoleLimits>,
) -> NativeExecutionResult<Option<Rc<runmat_runtime::context::PortableConsoleHost>>> {
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    controls
        .map(|controls| {
            controls
                .validate_transport(u64::from(limits.max_message_bytes))
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            let sink: Rc<dyn PortableConsoleSink> = Rc::new(FramedSink(limits));
            runmat_runtime::context::PortableConsoleHost::with_sink(controls, Some(sink))
                .map(Rc::new)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
        })
        .transpose()
}

/// Execute a reserved Direct program, streaming its requested console records.
pub(super) async fn execute_reserved(
    request: ReservedProgramExecutionRequest,
) -> NativeExecutionResult<(WorkerResponse, Option<ConsoleReceipt>)> {
    let Some(controls) = request.request().console else {
        return Ok((super::execute_reserved(request).await, None));
    };
    request
        .validate()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let limits = FrameLimits {
        max_message_bytes: WORKER_MAX_MESSAGE_BYTES,
    };
    controls
        .validate_transport(u64::from(limits.max_message_bytes))
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let sink: Rc<dyn PortableConsoleSink> = Rc::new(FramedSink(limits));
    Ok(crate::program_workload::execute_unit_with_console(
        request.request(),
        None,
        None,
        None,
        Some(sink),
    )
    .await)
}
