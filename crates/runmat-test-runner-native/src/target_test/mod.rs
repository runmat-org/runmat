//! Native target-test processes, distinct from the portable worker protocol.
//!
//! The selected linker supplies the image receipt. The receiver checks its
//! immutable compiled inventory; this host checks the exact image bytes,
//! framed response, request identity and process exit. Neither side treats
//! decoded identity fields alone as permission to execute a test.

mod image;
mod application;
pub use application::{run_compiled_native_application, NativePairedApplicationReport};
mod process;

pub use process::run_native_target_test;

/// Execute a test from the completed compiler/artifact/link pipeline. The
/// lower process owner still rechecks the opened image's exact digest.
pub async fn run_compiled_native_target_test(
    product: &runmat_aot::CompiledSelectedNativePair,
    request: &runmat_execution_artifact::target_test::NativeTargetTestRequest,
    limits: NativeTargetTestRunLimits,
) -> crate::NativeRunnerResult<NativeTargetTestProcessReport> {
    process::run_native_target_test(
        product.linked().tests(),
        product.linked().test_header(),
        request,
        limits,
    )
    .await
}

/// Host/deployment storage limits, independent of a test's writable artifact
/// quota. A test producing no files can still execute a nonempty image.
#[derive(Clone, Copy, Debug)]
pub struct NativeTargetTestRunLimits {
    pub max_image_bytes: u64,
    pub max_stderr_bytes: usize,
}

impl Default for NativeTargetTestRunLimits {
    fn default() -> Self {
        Self {
            max_image_bytes: 1024 * 1024 * 1024,
            max_stderr_bytes: 64 * 1024,
        }
    }
}

use runmat_execution_artifact::target_test::NativeTargetTestResult;
use runmat_test::result::TerminalDisposition;

/// A host failure has no invented receiver allocation measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeTargetTestHostFailure {
    Timeout,
    Process,
    Protocol,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeTargetTestCompletion {
    Receiver(NativeTargetTestResult),
    HostFailure {
        kind: NativeTargetTestHostFailure,
        diagnostic: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTargetTestProcessReport {
    pub completion: NativeTargetTestCompletion,
    /// Bounded console/diagnostic capture, separate from the protocol stream.
    pub stderr: String,
}

impl NativeTargetTestProcessReport {
    pub fn terminal_disposition(&self) -> TerminalDisposition {
        match &self.completion {
            NativeTargetTestCompletion::Receiver(result) => result.terminal_disposition(),
            NativeTargetTestCompletion::HostFailure { kind, .. } => match kind {
                NativeTargetTestHostFailure::Timeout => TerminalDisposition::TimedOut,
                NativeTargetTestHostFailure::Process => TerminalDisposition::Crashed,
                NativeTargetTestHostFailure::Protocol => TerminalDisposition::Failed,
            },
        }
    }
}
