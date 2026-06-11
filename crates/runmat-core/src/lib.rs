#![allow(clippy::result_large_err)]

#[cfg(all(test, target_arch = "wasm32"))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

pub mod abi;
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

/// Test-only helper that executes a text source via `ExecutionRequest`.
#[cfg(not(target_arch = "wasm32"))]
pub fn execute_text_request_for_testing(
    session: &mut RunMatSession,
    source_text: &str,
) -> Result<SessionExecutionResult, RunError> {
    let request = abi::ExecutionRequest::for_source(
        abi::SourceInput::Text {
            name: "<test>".to_string(),
            text: source_text.to_string(),
        },
        session.compat_mode(),
        abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );
    let response = futures::executor::block_on(session.execute_request(request));
    let outcome = response.result?;
    let workspace = session.workspace_snapshot();
    let warnings = outcome
        .diagnostics
        .iter()
        .filter(|diag| diag.severity == abi::DiagnosticSeverity::Warning)
        .map(|diag| runmat_runtime::warning_store::RuntimeWarning {
            identifier: diag.code.clone(),
            message: diag.message.clone(),
        })
        .collect();
    let error = outcome
        .diagnostics
        .iter()
        .find(|diag| diag.severity == abi::DiagnosticSeverity::Error)
        .map(|diag| {
            runmat_runtime::build_runtime_error(diag.message.clone())
                .with_identifier(diag.code.clone())
                .with_call_stack(diag.callstack.clone())
                .with_call_frames_elided(diag.callstack_elided)
                .build()
        });
    Ok(SessionExecutionResult {
        value: outcome.flow.durable_workspace_value().cloned(),
        execution_time_ms: outcome.execution_time_ms,
        used_jit: outcome.used_jit,
        error,
        type_info: outcome.type_info,
        streams: outcome.streams,
        workspace,
        figures_touched: outcome.figures_touched,
        warnings,
        profiling: outcome.profiling,
        fusion_plan: outcome.fusion_plan,
        stdin_events: outcome.stdin_events,
    })
}
