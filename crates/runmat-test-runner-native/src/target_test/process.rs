use std::time::Duration;

use runmat_aot::link::LinkedProgram;
use runmat_execution_artifact::target_test::{
    NativePairedProductHeader, NativeTargetTestRequest, NativeTargetTestResult,
    MAX_NATIVE_TARGET_TEST_WIRE_BYTES,
};
use runmat_process_host::child::BootstrapBoundChild;
use runmat_process_host::ipc::frame::{read_payload, write_payload, FrameLimits};
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
    let child = image
        .command
        .clone()
        .spawn_observed(&image.spawn)
        .await
        .map_err(|error| NativeRunnerError::Configuration(error.to_string()))?;
    let mut child = BootstrapBoundChild::new(child, image.take_lease(), image.spawn.clone());
    let stderr = child.captured_stderr();
    let mut stdio = child
        .take_stdio()
        .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
    let limits = FrameLimits {
        max_message_bytes: MAX_NATIVE_TARGET_TEST_WIRE_BYTES as u32,
    };
    let deadline = Duration::from_millis(request.resources.max_wall_millis);
    let exchange = async {
        write_payload(&mut stdio.stdin, &payload, limits)
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        stdio
            .stdin
            .shutdown()
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let response = read_payload(&mut stdio.stdout, limits)
            .await
            .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let result =
            NativeTargetTestResult::from_canonical_bytes(&response, request, &header.binding)
                .map_err(|error| failure(NativeTargetTestHostFailure::Protocol, error))?;
        let mut extra = [0_u8; 1];
        if stdio
            .stdout
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
    let completion = match tokio::time::timeout(deadline, exchange).await {
        Ok(Ok(result)) => NativeTargetTestCompletion::Receiver(result),
        Ok(Err(error)) => error,
        Err(_) => failure(
            NativeTargetTestHostFailure::Timeout,
            "native test process exceeded its deadline",
        ),
    };
    // Reaping the direct child alone does not retire descendants that inherited
    // process authority. The common owner also handles cancellation/unwind.
    child.terminate_contained_child().await.map_err(|error| {
        NativeRunnerError::Protocol(format!("native test process retirement failed: {error}"))
    })?;
    Ok(NativeTargetTestProcessReport {
        completion,
        stderr: stderr.text(),
    })
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
