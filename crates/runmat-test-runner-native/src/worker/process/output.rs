//! Transfer raw output only after the process tree's retirement is proved.
use std::time::Duration;

use runmat_execution::process_output::{
    ProcessOutputBytes, ProcessStreamCompletion, ProcessStreamOutput,
};
use runmat_process_host::{CapturedProcessOutput, ProcessOutputCompletion, ProcessOutputSnapshot};
use runmat_test::event::{WorkerProcessOutput, WorkerProcessRetirement};

use super::*;

pub(super) async fn take(
    session: &ProcessSession,
) -> Result<Option<WorkerProcessOutput>, BackendError> {
    let retired = session.state.retired.load(Ordering::Acquire);
    if retired {
        // No ownership is claimed before this await: cancellation can retry.
        // A stalled producer is reported as incomplete, never certified as EOF.
        let _ = tokio::time::timeout(Duration::from_secs(30), async {
            tokio::join!(
                session.state.stdout.wait_closed(),
                session.state.stderr.wait_closed()
            );
        })
        .await;
    }
    if session.state.output_taken.swap(true, Ordering::AcqRel) {
        return Ok(None);
    }
    let stdout = snapshot(&session.state.stdout)?;
    let stderr = snapshot(&session.state.stderr)?;
    if retired
        && stdout.observed_bytes == Some(0)
        && stderr.observed_bytes == Some(0)
        && stdout.completion == ProcessOutputCompletion::EndOfStream
        && stderr.completion == ProcessOutputCompletion::EndOfStream
    {
        return Ok(None);
    }
    Ok(Some(WorkerProcessOutput {
        worker: session.id.clone(),
        retirement: if retired {
            WorkerProcessRetirement::Verified
        } else {
            WorkerProcessRetirement::Unverified
        },
        stdout: stream(stdout),
        stderr: stream(stderr),
    }))
}

fn snapshot(capture: &CapturedProcessOutput) -> Result<ProcessOutputSnapshot, BackendError> {
    match capture.completion() {
        ProcessOutputCompletion::EndOfStream | ProcessOutputCompletion::ReadFailed(_) => {
            capture.take_closed_snapshot().map_err(transport_error)
        }
        // Partial diagnostics keep the live producer's prefix in place.
        ProcessOutputCompletion::NotCaptured
        | ProcessOutputCompletion::Reading
        | ProcessOutputCompletion::Transferred => Ok(capture.snapshot()),
    }
}

fn stream(snapshot: ProcessOutputSnapshot) -> ProcessStreamOutput {
    let completion = match snapshot.completion {
        ProcessOutputCompletion::EndOfStream => ProcessStreamCompletion::EndOfStream,
        ProcessOutputCompletion::ReadFailed(kind) => ProcessStreamCompletion::ReadFailed {
            kind: format!("{kind:?}"),
        },
        ProcessOutputCompletion::NotCaptured => ProcessStreamCompletion::NotCaptured,
        ProcessOutputCompletion::Reading => ProcessStreamCompletion::Reading,
        ProcessOutputCompletion::Transferred => ProcessStreamCompletion::Transferred,
    };
    ProcessStreamOutput {
        bytes: ProcessOutputBytes::new(snapshot.bytes.into_vec()),
        observed_bytes: snapshot.observed_bytes,
        truncated: snapshot.truncated,
        completion,
    }
}
