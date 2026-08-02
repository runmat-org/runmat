use runmat_test_runner::reporter::{
    HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::Coordinator;
use wasm_bindgen::prelude::*;

use crate::wire::errors::js_error;

use super::backend::JsWorkerBackend;
use super::clock::{BrowserCancellation, BrowserClock};
use super::wire::{BrowserReport, BrowserRunInput, BrowserRunOutput};

/// Run a complete immutable test plan through the portable Rust coordinator.
/// JavaScript owns only worker construction, transport, termination, and host
/// storage; it does not implement scheduling or lifecycle policy.
#[wasm_bindgen(js_name = runTests)]
pub async fn run_tests(input: JsValue, backend: JsValue) -> Result<JsValue, JsValue> {
    let input: BrowserRunInput = serde_wasm_bindgen::from_value(input)
        .map_err(|error| js_error(&format!("Browser test input is invalid: {error}")))?;
    let report_formats = input.options.reports.clone();
    let (submission, config) = input
        .into_parts()
        .map_err(|error| js_error(&error.to_string()))?;
    let coordinator = Coordinator::new(config).map_err(|error| js_error(&error.to_string()))?;
    let worker_backend =
        JsWorkerBackend::new(backend.clone()).map_err(|error| js_error(&error.to_string()))?;
    let cancellation = BrowserCancellation::new(backend);
    let mut reporters = ReporterFanout::default();
    for report in report_formats {
        match report {
            BrowserReport::Human => reporters.push(HumanReporter::default()),
            BrowserReport::Json => reporters.push(JsonReporter::default()),
            BrowserReport::Junit => reporters.push(JunitReporter),
            BrowserReport::Tap => reporters.push(TapReporter),
        }
    }
    let run = coordinator
        .run(
            submission,
            &worker_backend,
            &BrowserClock,
            &cancellation,
            &NoopTelemetry,
            &mut reporters,
        )
        .await
        .map_err(|error| js_error(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&BrowserRunOutput {
        result: run.result,
        events: run.events,
        reports: run.reports.into_iter().map(Into::into).collect(),
        infrastructure_failures: run.infrastructure_failures,
        isolation: run.isolation,
    })
    .map_err(|error| {
        js_error(&format!(
            "Browser test result serialization failed: {error}"
        ))
    })
}
