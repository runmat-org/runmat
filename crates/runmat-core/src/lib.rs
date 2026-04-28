#![allow(clippy::result_large_err)]

use runmat_lexer::tokenize_detailed;

#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

mod error;
mod execution;
mod fusion;
mod profiling;
mod session;
mod source_pool;
mod telemetry;
mod value_metadata;
mod workspace;

pub use error::{runtime_error_telemetry_failure_info, RunError};
pub use execution::*;
pub use fusion::*;
pub use runmat_parser::CompatMode;
pub use session::RunMatSession;
pub use telemetry::{
    TelemetryFailureInfo, TelemetryHost, TelemetryPlatformInfo, TelemetryRunConfig,
    TelemetryRunFinish, TelemetryRunGuard, TelemetrySink,
};
pub use value_metadata::{
    approximate_size_bytes, matlab_class_name, numeric_dtype_label, preview_numeric_values,
    value_shape,
};
pub use workspace::*;

#[cfg(test)]
mod tests;

/// Tokenize the input string and return a space separated string of token names.
/// This is kept for backward compatibility with existing tests.
pub fn format_tokens(input: &str) -> String {
    tokenize_detailed(input)
        .into_iter()
        .map(|t| format!("{:?}", t.token))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Execute MATLAB/Octave code and return the result as a formatted string
pub async fn execute_and_format(input: &str) -> String {
    match RunMatSession::new() {
        Ok(mut engine) => match engine.execute(input).await {
            Ok(result) => {
                if let Some(error) = result.error {
                    format!("Error: {error}")
                } else if let Some(value) = result.value {
                    format!("{value:?}")
                } else {
                    "".to_string()
                }
            }
            Err(e) => format!("Error: {e}"),
        },
        Err(e) => format!("Engine Error: {e}"),
    }
}
