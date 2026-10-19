use super::{NativeProcessProvider, ProcessControl, TransportMeters};
use crate::protocol::{WorkerLaunchRequest, WorkerProcessMessage, WorkerResponse};
use runmat_process_host::{
    ipc::{read_payload, write_payload, FrameLimits, MeteredIo},
    ChildProcess,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::io::BufReader;

pub(super) async fn exchange(
    provider: &NativeProcessProvider<'_>,
    child: &mut ChildProcess,
    request: &WorkerLaunchRequest,
    control: &dyn ProcessControl,
    meters: &TransportMeters,
    console: &mut Option<runmat_execution::console::ConsoleTranscript>,
) -> Result<WorkerResponse, String> {
    let stdio = child.take_stdio().map_err(|error| error.to_string())?;
    let mut reader = BufReader::new(MeteredIo::new(stdio.stdout, meters.received.clone()));
    let mut writer = MeteredIo::new(stdio.stdin, meters.sent.clone());
    let limits = FrameLimits {
        max_message_bytes: provider.config.max_message_bytes,
    };
    let payload =
        runmat_canonical_codec::encode_json_bounded(request, u64::from(limits.max_message_bytes))
            .map_err(|error| error.to_string())?;
    write_payload(&mut writer, &payload, limits)
        .await
        .map_err(|error| error.to_string())?;
    let WorkerLaunchRequest::Reserved(reserved) = request else {
        return Err("durable program provider launches reserved direct programs only".into());
    };
    let program = reserved.request();
    let mut receiver = program
        .console
        .map(runmat_execution::console::ConsoleReceiver::new)
        .transpose()
        .map_err(|error| error.to_string())?;
    let mut receipt = None;
    let response = read_response(
        &mut reader,
        limits,
        program,
        control,
        &mut receiver,
        &mut receipt,
    )
    .await;
    *console = receiver
        .map(|receiver| receiver.finish(receipt.as_ref()))
        .transpose()
        .map_err(|error| error.to_string())?;
    let response = response?;
    // Keep the request pipe open until a terminal frame is received. EOF before
    // then is parent death to domain workers, and would cancel healthy execution.
    drop(writer);
    response
        .validate_against(program)
        .map_err(|error| error.to_string())?;
    Ok(response)
}

async fn read_response(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    limits: FrameLimits,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    control: &dyn ProcessControl,
    receiver: &mut Option<runmat_execution::console::ConsoleReceiver>,
    receipt: &mut Option<runmat_execution::console::ConsoleReceipt>,
) -> Result<WorkerResponse, String> {
    let mut last_progress_sequence = 0;
    loop {
        let payload = read_payload_until_cancelled(reader, limits, control.cancellation())
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "execution was cancelled".to_string())?;
        match serde_json::from_slice::<WorkerProcessMessage>(&payload)
            .map_err(|error| error.to_string())?
        {
            WorkerProcessMessage::Progress { progress } => {
                progress.validate()?;
                if progress.sequence <= last_progress_sequence {
                    return Err("native worker progress is not strictly monotone".into());
                }
                last_progress_sequence = progress.sequence;
                control.record_progress(progress);
            }
            WorkerProcessMessage::Console { record } => {
                let receiver = receiver
                    .as_mut()
                    .ok_or("worker emitted unrequested console evidence")?;
                receiver.push(record).map_err(|error| error.to_string())?;
                control
                    .record_console(receiver.records().last().expect("retained original record"))?;
            }
            WorkerProcessMessage::Completed { response, console } => {
                match (receiver.as_ref(), console.as_ref()) {
                    (Some(receiver), Some(console)) => receiver
                        .verify_receipt(console)
                        .map_err(|error| error.to_string())?,
                    (None, None) => {}
                    _ => {
                        return Err("worker console receipt differs from requested controls".into())
                    }
                }
                response
                    .validate_against(request)
                    .map_err(|error| error.to_string())?;
                if let Some(console) = &console {
                    control.finish_console(console)?;
                }
                *receipt = console;
                return Ok(response);
            }
            WorkerProcessMessage::CollectiveRequest { .. } => {
                return Err("durable program attempts do not serve worker collectives".into());
            }
            WorkerProcessMessage::InfrastructureFailure { message }
            | WorkerProcessMessage::RetirementUnverified { message } => return Err(message),
        }
    }
}

pub(crate) async fn read_payload_until_cancelled(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    limits: FrameLimits,
    cancelled: &AtomicBool,
) -> runmat_process_host::ProcessHostResult<Option<Vec<u8>>> {
    // A framed read may span several pipe reads. Keep the same future alive
    // while polling cancellation: recreating it after a timer tick discards
    // bytes already consumed from the length-prefixed stream.
    let read = read_payload(reader, limits);
    tokio::pin!(read);
    loop {
        tokio::select! {
            response = &mut read => return response.map(Some),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(None);
                }
            }
        }
    }
}
