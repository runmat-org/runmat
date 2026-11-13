//! Private process modes shared by application and target-test images.

use std::ffi::OsString;

use runmat_process_host::{HiddenMode, HiddenModeRegistry};

use crate::execute::ProcessFailure;

/// Dispatch before reading any user-program or test-request payload. Extension
/// hosts authenticate through their own bootstrap; this is not test stdin.
pub(super) fn run_if_requested(arguments: Vec<OsString>) -> Result<bool, ProcessFailure> {
    match HiddenModeRegistry::standard()
        .detect(arguments)
        .map_err(|error| error.to_string())?
    {
        None => Ok(false),
        Some(HiddenMode::ExtensionHost) => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|error| {
                    ProcessFailure::Runtime(format!(
                        "standalone extension-host runtime failed to start: {error}"
                    ))
                })?;
            runtime
                .block_on(runmat_runtime::foreign::run_extension_host())
                .map_err(|error| {
                    ProcessFailure::Runtime(format!("standalone extension host failed: {error}"))
                })?;
            Ok(true)
        }
        Some(mode) => Err(format!(
            "standalone executable does not provide private host mode '{}'",
            mode.marker()
        )
        .into()),
    }
}
