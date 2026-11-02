use std::time::Duration;

use runmat_aot::link::LinkedProgram;
use runmat_execution_artifact::target_test::{
    NativePairedProductHeader, NativeTargetTestRequest, NativeTargetTestResult,
    MAX_NATIVE_TARGET_TEST_WIRE_BYTES,
};
use runmat_process_host::child::BootstrapBoundChild;
use runmat_process_host::ipc::frame::{read_payload, write_payload, FrameLimits};
use runmat_process_host::ipc::private_control::PrivateControlListener;
use runmat_process_host::{
    CapturedProcessOutput, ProcessOutputCompletion, ProcessOutputSnapshot, StdioPolicy,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

use super::image::PendingImage;
use super::{
    NativeTargetTestCompletion, NativeTargetTestHostFailure, NativeTargetTestProcessReport,
    NativeTargetTestRunLimits,
};
use crate::{NativeRunnerError, NativeRunnerResult};

/// Run one request in one linked test process. A terminal language/test failure
/// is reported in a valid response with a successful protocol-process exit.
/// Timeout/crash/protocol errors are host observations, not receiver results.
pub async fn run_native_target_test(
    linked: &LinkedProgram,
    header: &NativePairedProductHeader,
    request: &NativeTargetTestRequest,
    limits: NativeTargetTestRunLimits,
) -> NativeRunnerResult<NativeTargetTestProcessReport> {
    let payload = request
        .to_canonical_bytes(&header.binding)
        .map_err(|error| NativeRunnerError::Configuration(error.to_string()))?;
    let mut image = PendingImage::prepare(linked, header, limits)?;
    let listener = PrivateControlListener::bind()
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    image.command.stdio = StdioPolicy::PrivateControl(listener.endpoint().clone());
    let child = image
        .command
        .clone()
        .spawn_observed(&image.spawn)
        .await
        .map_err(|error| NativeRunnerError::Configuration(error.to_string()))?;
    let mut child = BootstrapBoundChild::new(child, image.take_lease(), image.spawn.clone());
    let stderr = child.captured_stderr();
    let stdout = CapturedProcessOutput::new(limits.max_stdout_bytes);
    let limits = FrameLimits {
        max_message_bytes: MAX_NATIVE_TARGET_TEST_WIRE_BYTES as u32,
    };
    let deadline = Duration::from_millis(request.resources.max_wall_millis);
    let exchange = async {
        let application = child
            .take_application_output()
            .map_err(|error| failure(NativeTargetTestHostFailure::Process, error))?;
        stdout
            .drain(application.stdout)
            .map_err(|error| failure(NativeTargetTestHostFailure::Process, error))?;
        application.stderr_diagnostics.drain(application.stderr);
        // Bootstrap is inside the same wall-time bound as request execution.
        let mut control = listener
            .accept()
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        write_payload(&mut control.writer, &payload, limits)
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        control
            .writer
            .shutdown()
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let response = read_payload(&mut control.reader, limits)
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let result =
            NativeTargetTestResult::from_canonical_bytes(&response, request, &header.binding)
                .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let mut extra = [0_u8; 1];
        if control
            .reader
            .read(&mut extra)
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?
            != 0
        {
            return Err(failure(
                NativeTargetTestHostFailure::Protocol,
                "trailing protocol output",
            ));
        }
        let exit = child
            .wait_and_retire_tree()
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Process, error))?;
        if !exit.success {
            return Err(failure(
                NativeTargetTestHostFailure::Process,
                format!(
                    "native test process exited with code {:?}, signal {:?}",
                    exit.code, exit.signal
                ),
            ));
        }
        Ok(result)
    };
    let mut completion = match tokio::time::timeout(deadline, exchange).await {
        Ok(Ok(result)) => NativeTargetTestCompletion::Receiver(result),
        Ok(Err(error)) => error,
        Err(_) => failure(
            NativeTargetTestHostFailure::Timeout,
            "native test process exceeded its deadline",
        ),
    };
    // Reaping the direct child alone does not retire descendants that inherited
    // process authority. The common owner also handles cancellation/unwind.
    if let Err(error) = child.terminate_contained_child().await {
        completion = NativeTargetTestCompletion::RetirementUnverified {
            original: Box::new(completion),
            diagnostic: error.to_string(),
        };
    }
    // Pipe EOF is independent of receiving a control result and of reaping the
    // direct child. Do not label a timed-out drain complete or discard its prefix.
    let _ = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(stdout.wait_closed(), stderr.wait_closed());
    })
    .await;
    Ok(NativeTargetTestProcessReport {
        completion,
        stdout: output_snapshot(&stdout)?,
        stderr: output_snapshot(&stderr)?,
    })
}

fn output_snapshot(capture: &CapturedProcessOutput) -> NativeRunnerResult<ProcessOutputSnapshot> {
    match capture.completion() {
        ProcessOutputCompletion::EndOfStream | ProcessOutputCompletion::ReadFailed(_) => capture
            .take_closed_snapshot()
            .map_err(|error| NativeRunnerError::Protocol(error.to_string())),
        ProcessOutputCompletion::NotCaptured
        | ProcessOutputCompletion::Reading
        | ProcessOutputCompletion::Transferred => Ok(capture.snapshot()),
    }
}

fn failure(
    kind: NativeTargetTestHostFailure,
    diagnostic: impl std::fmt::Display,
) -> NativeTargetTestCompletion {
    NativeTargetTestCompletion::HostFailure {
        kind,
        diagnostic: diagnostic.to_string(),
    }
}
