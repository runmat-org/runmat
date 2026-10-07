//! Source capture uses the existing runtime host; stdout stays a framed protocol.
use super::*;
use crate::protocol::WorkerProcessMessage;
use runmat_execution::console::ConsoleRecord;
use runmat_runtime::context::PortableConsoleSink;
use std::rc::Rc;

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
        runmat_process_host::ipc::write_payload_sync(&mut std::io::stdout().lock(), &bytes, self.0)
            .map_err(|error| error.to_string())
    }
}

pub(super) async fn execute_streamed(
    request: WorkerRequest,
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    limits: FrameLimits,
    parent: &runmat_process_host::ipc::WorkerParentGuard,
) -> NativeExecutionResult<()> {
    let Some(controls) = request.console else {
        let response = execute(request).await;
        parent.disarm();
        return write_message(
            writer,
            &WorkerProcessMessage::Completed {
                response,
                console: None,
            },
            limits,
        )
        .await;
    };
    request
        .validate_for_portable_host()
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    controls
        .validate_transport(u64::from(limits.max_message_bytes))
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let sink: Rc<dyn PortableConsoleSink> = Rc::new(FramedSink(limits));
    let (response, console) =
        crate::program_workload::execute_unit_with_console(&request, None, None, None, Some(sink))
            .await;
    parent.disarm();
    write_message(
        writer,
        &WorkerProcessMessage::Completed { response, console },
        limits,
    )
    .await
}

async fn write_message(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    message: &WorkerProcessMessage,
    limits: FrameLimits,
) -> NativeExecutionResult<()> {
    let bytes =
        runmat_canonical_codec::encode_json_bounded(message, u64::from(limits.max_message_bytes))?;
    write_payload(writer, &bytes, limits).await?;
    Ok(())
}
