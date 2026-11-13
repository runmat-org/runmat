use super::{NativeProcessProvider, ProcessControl, TransportMeters};
use crate::protocol::{WorkerLaunchRequest, WorkerProcessMessage, WorkerResponse};
use runmat_process_host::ipc::{
    private_control::PrivateControlListener, read_payload, write_payload, FrameLimits, MeteredIo,
};
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::io::BufReader;

pub(super) async fn exchange(
    provider: &NativeProcessProvider<'_>,
    listener: PrivateControlListener,
    request: &WorkerLaunchRequest,
    control: &dyn ProcessControl,
    meters: &TransportMeters,
    console: &mut Option<runmat_execution::console::ConsoleTranscript>,
    console_scopes: &mut Vec<runmat_execution::console::ScopedConsoleTranscript>,
) -> Result<WorkerResponse, String> {
    // Both application pipes are already owned by the sibling drain future.
    let accept = tokio::time::timeout(Duration::from_secs(30), listener.accept());
    tokio::pin!(accept);
    let connected = loop {
        tokio::select! {
            connected = &mut accept => break connected
                .map_err(|_| "native worker private control bootstrap exceeded its deadline".to_string())?
                .map_err(|error| error.to_string())?,
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if should_stop(control) {
                    return Err(stop_diagnostic(control));
                }
            }
        }
    };
    if should_stop(control) {
        return Err(stop_diagnostic(control));
    }
    let mut reader = BufReader::new(MeteredIo::new(connected.reader, meters.received.clone()));
    let mut writer = MeteredIo::new(connected.writer, meters.sent.clone());
    let limits = FrameLimits {
        max_message_bytes: provider.config.max_message_bytes,
    };
    let payload =
        runmat_canonical_codec::encode_json_bounded(request, u64::from(limits.max_message_bytes))
            .map_err(|error| error.to_string())?;
    {
        let write = write_payload(&mut writer, &payload, limits);
        tokio::pin!(write);
        loop {
            tokio::select! {
                result = &mut write => {
                    result.map_err(|error| error.to_string())?;
                    break;
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if should_stop(control) {
                        return Err(stop_diagnostic(control));
                    }
                }
            }
        }
    }
    if should_stop(control) {
        return Err(stop_diagnostic(control));
    }
    let reserved = match request {
        WorkerLaunchRequest::Reserved(reserved)
        | WorkerLaunchRequest::ReservedConsole {
            request: reserved, ..
        } => reserved,
        _ => return Err("durable program provider launches reserved direct programs only".into()),
    };
    let program = reserved.request();
    let mut receiver = program
        .console
        .map(runmat_execution::console::ConsoleReceiver::new)
        .transpose()
        .map_err(|error| error.to_string())?;
    let mut receipt = None;
    let mut received_scopes = Vec::new();
    let response = read_response(
        &mut reader,
        limits,
        program,
        control,
        &mut receiver,
        &mut receipt,
        &mut received_scopes,
    )
    .await;
    *console = receiver
        .map(|receiver| receiver.finish(receipt.as_ref()))
        .transpose()
        .map_err(|error| error.to_string())?;
    crate::console_bridge::validate_imported_subtree(
        console.as_ref(),
        &received_scopes,
        request.console_budget(),
    )?;
    *console_scopes = received_scopes;
    let response = response?;
    // Keep the request pipe open until a terminal frame is received. EOF before
    // then is parent death to domain workers, and would cancel healthy execution.
    drop(writer);
    response
        .validate_against(program)
        .map_err(|error| error.to_string())?;
    Ok(response)
}

fn stop_diagnostic(control: &dyn ProcessControl) -> String {
    control
        .failure()
        .or_else(|| control.raw_output_failure().map(|error| error.to_string()))
        .unwrap_or_else(|| {
            if control.cancellation_reason().is_some() {
                "execution was cancelled".into()
            } else {
                "execution control stopped without an authenticated cancellation reason".into()
            }
        })
}

fn should_stop(control: &dyn ProcessControl) -> bool {
    control.cancellation().load(Ordering::Acquire) || control.raw_output_failure().is_some()
}

async fn read_response(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    limits: FrameLimits,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    control: &dyn ProcessControl,
    receiver: &mut Option<runmat_execution::console::ConsoleReceiver>,
    receipt: &mut Option<runmat_execution::console::ConsoleReceipt>,
    console_scopes: &mut Vec<runmat_execution::console::ScopedConsoleTranscript>,
) -> Result<WorkerResponse, String> {
    let mut last_progress_sequence = 0;
    loop {
        let payload = read_payload_until_stopped(reader, limits, || should_stop(control))
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| stop_diagnostic(control))?;
        match serde_json::from_slice::<WorkerProcessMessage>(&payload)
            .map_err(|error| error.to_string())?
        {
            WorkerProcessMessage::FeaAdmissionCall { .. } => {
                return Err("durable worker requested ungranted FEA proxy".into())
            }
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
            WorkerProcessMessage::Completed {
                response,
                console,
                console_scopes: scopes,
            } => {
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
                *console_scopes = scopes;
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

pub(crate) async fn read_payload_until_stopped(
    reader: &mut (impl tokio::io::AsyncRead + Unpin),
    limits: FrameLimits,
    should_stop: impl Fn() -> bool,
) -> runmat_process_host::ProcessHostResult<Option<Vec<u8>>> {
    // Keep this one read future alive across stop ticks: a new future would
    // discard a partial length prefix or body already consumed from the pipe.
    let read = read_payload(reader, limits);
    tokio::pin!(read);
    loop {
        tokio::select! {
            response = &mut read => return response.map(Some),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                if should_stop() { return Ok(None); }
            }
        }
    }
}
