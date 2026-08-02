use runmat_test_runner::coverage::CoverageReportFormat;
use runmat_test_runner::plugin::{CoveragePlugin, PluginFanout};
use runmat_test_runner::reporter::{
    HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::Coordinator;
use wasm_bindgen::prelude::*;

use crate::wire::errors::js_error;

use super::backend::JsWorkerBackend;
use super::clock::{BrowserCancellation, BrowserClock};
use super::wire::{BrowserCoverageFormat, BrowserReport, BrowserRunInput, BrowserRunOutput};

/// Run a complete immutable test plan through the portable Rust coordinator.
/// JavaScript owns only worker construction, transport, termination, and host
/// storage; it does not implement scheduling or lifecycle policy.
#[wasm_bindgen(js_name = runTests)]
pub async fn run_tests(input: JsValue, backend: JsValue) -> Result<JsValue, JsValue> {
    let input: BrowserRunInput = serde_wasm_bindgen::from_value(input)
        .map_err(|error| js_error(&format!("Browser test input is invalid: {error}")))?;
    let report_formats = input.options.reports.clone();
    let coverage_options = input.options.coverage.clone();
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
    let mut run = coordinator
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
    if coverage_options.is_requested() {
        let formats = if coverage_options.formats.is_empty() {
            vec![CoverageReportFormat::Json, CoverageReportFormat::Html]
        } else {
            coverage_options
                .formats
                .into_iter()
                .map(|format| match format {
                    BrowserCoverageFormat::Json => CoverageReportFormat::Json,
                    BrowserCoverageFormat::Lcov => CoverageReportFormat::Lcov,
                    BrowserCoverageFormat::Cobertura => CoverageReportFormat::Cobertura,
                    BrowserCoverageFormat::Html => CoverageReportFormat::Html,
                })
                .collect()
        };
        let filter = runmat_test::coverage::CoverageFilter {
            roots: coverage_options.roots,
            exclude: coverage_options.exclude,
            include_generated: coverage_options.include_generated,
            include_vendor: coverage_options.include_vendor,
        };
        let mut plugins = PluginFanout::default();
        plugins.push(CoveragePlugin::new(filter, formats));
        plugins.apply(&mut run);
    }
    let coverage = run.coverage.clone();
    serde_wasm_bindgen::to_value(&BrowserRunOutput {
        result: run.result,
        events: run.events,
        reports: run.reports.into_iter().map(Into::into).collect(),
        infrastructure_failures: run.infrastructure_failures,
        plugin_failures: run.plugin_failures,
        isolation: run.isolation,
        coverage,
    })
    .map_err(|error| {
        js_error(&format!(
            "Browser test result serialization failed: {error}"
        ))
    })
}
