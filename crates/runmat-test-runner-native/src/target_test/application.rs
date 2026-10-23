//! Execute the application half through the same checked private-image owner.

use std::time::Duration;

use runmat_execution_artifact::target_test::NativePairedProductRole;
use runmat_process_host::child::BootstrapBoundChild;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

use super::{image::PendingImage, NativeTargetTestRunLimits};
use crate::{NativeRunnerError, NativeRunnerResult};

#[derive(Debug)]
pub struct NativePairedApplicationReport {
    pub exit: runmat_process_host::ProcessExit,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

/// Rehash and privately stage the completed application bytes before launch.
/// Console capture and wall time are explicitly bounded; they are not a
/// managed-memory measurement or a deployment-service grant.
pub async fn run_compiled_native_application(
    product: &runmat_aot::CompiledSelectedNativePair,
    limits: NativeTargetTestRunLimits,
    max_stdout_bytes: usize,
    max_wall: Duration,
) -> NativeRunnerResult<NativePairedApplicationReport> {
    run_linked_native_application(product.linked(), limits, max_stdout_bytes, max_wall).await
}

/// Execute an independently reopened physical pair through the same private
/// image validator and process retirement owner as a newly compiled pair.
pub async fn run_linked_native_application(
    product: &runmat_aot::link::LinkedNativePair,
    limits: NativeTargetTestRunLimits,
    max_stdout_bytes: usize,
    max_wall: Duration,
) -> NativeRunnerResult<NativePairedApplicationReport> {
    if max_stdout_bytes == 0 || max_wall.is_zero() {
        return Err(configuration(
            "application output and time bounds must be nonzero",
        ));
    }
    let mut image = PendingImage::prepare_role(
        product.application(),
        product.application_header(),
        NativePairedProductRole::Application,
        limits,
    )?;
    let child = image
        .command
        .clone()
        .spawn_observed(&image.spawn)
        .await
        .map_err(configuration)?;
    let mut child = BootstrapBoundChild::new(child, image.take_lease(), image.spawn.clone());
    let stderr = child.captured_stderr();
    let result = async {
        let mut stdio = child.take_stdio().map_err(configuration)?;
        stdio.stdin.shutdown().await?;
        let mut stdout = Vec::new();
        stdio
            .stdout
            .take((max_stdout_bytes as u64).saturating_add(1))
            .read_to_end(&mut stdout)
            .await?;
        if stdout.len() > max_stdout_bytes {
            return Err(configuration("application console exceeded its byte bound"));
        }
        let exit = child.wait_and_retire_tree().await.map_err(configuration)?;
        Ok(NativePairedApplicationReport {
            exit,
            stdout,
            stderr: stderr.text(),
        })
    };
    let result = match tokio::time::timeout(max_wall, result).await {
        Ok(result) => result,
        Err(_) => Err(configuration("application exceeded its wall-time bound")),
    };
    // Keep the contained process and image lease until physical retirement,
    // including output overflow, timeout, and failed process wait.
    child
        .terminate_contained_child()
        .await
        .map_err(configuration)?;
    result
}

fn configuration(error: impl std::fmt::Display) -> NativeRunnerError {
    NativeRunnerError::Configuration(error.to_string())
}
