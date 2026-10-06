use super::{NativeProcessProvider, ProcessControl, TransportMeters};
use crate::protocol::{WorkerProcessMessage, WorkerRequest, WorkerResponse};
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
    request: &WorkerRequest,
    control: &dyn ProcessControl,
    meters: &TransportMeters,
) -> Result<WorkerResponse, String> {
    let stdio = child.take_stdio().map_err(|error| error.to_string())?;
    let mut reader = BufReader::new(MeteredIo::new(stdio.stdout, meters.received.clone()));
    let mut writer = MeteredIo::new(stdio.stdin, meters.sent.clone());
    let limits = FrameLimits {
        max_message_bytes: provider.config.max_message_bytes,
    };
    let payload = serde_json::to_vec(request).map_err(|error| error.to_string())?;
    write_payload(&mut writer, &payload, limits)
        .await
        .map_err(|error| error.to_string())?;
    let mut last_progress_sequence = 0;
    let response = loop {
        let payload = read_payload_until_cancelled(&mut reader, limits, control.cancellation())
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
            WorkerProcessMessage::Completed { response } => break response,
        }
    };
    // Keep the request pipe open until a terminal frame is received. EOF before
    // then is parent death to domain workers, and would cancel healthy execution.
    drop(writer);
    response
        .validate_against(request)
        .map_err(|error| error.to_string())?;
    Ok(response)
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
