use runmat_test_runner::coverage::CoverageReportFormat;
use runmat_test_runner::plugin::{CoveragePlugin, PluginFanout};
use runmat_test_runner::reporter::{
    EventObserver, HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::Coordinator;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::wire::errors::js_error;

use super::backend::JsWorkerBackend;
use super::clock::{BrowserCancellation, BrowserClock};
use super::wire::{BrowserCoverageFormat, BrowserReport, BrowserRunInput, BrowserRunOutput};

/// Run a complete immutable test plan through the portable Rust coordinator.
/// JavaScript owns only worker construction, transport, termination, and host
/// storage; it does not implement scheduling or lifecycle policy.
#[wasm_bindgen(js_name = runTests)]
pub async fn run_tests(input: JsValue, backend: JsValue) -> Result<JsValue, JsValue> {
    run_tests_inner(input, backend, None).await
}

/// Run tests and project each canonical coordinator event to JavaScript as it
/// occurs. The callback is a projection only; scheduling, lifecycle, results,
/// and report construction remain owned by the portable coordinator.
#[wasm_bindgen(js_name = runTestsWithEvents)]
pub async fn run_tests_with_events(
    input: JsValue,
    backend: JsValue,
    observer: JsValue,
) -> Result<JsValue, JsValue> {
    let observer = observer
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_error("Browser test event observer must be a function"))?;
    run_tests_inner(input, backend, Some(observer)).await
}

async fn run_tests_inner(
    input: JsValue,
    backend: JsValue,
    observer: Option<js_sys::Function>,
) -> Result<JsValue, JsValue> {
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
    if let Some(observer) = observer {
        reporters.push_observer(JsEventObserver(observer));
    }
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

struct JsEventObserver(js_sys::Function);

impl EventObserver for JsEventObserver {
    fn event(
        &mut self,
        event: &runmat_test::event::TestEvent,
    ) -> runmat_test_runner::RunnerResult<()> {
        let event = serde_wasm_bindgen::to_value(event).map_err(|error| {
            runmat_test_runner::RunnerError::Reporter(format!(
                "failed to serialize browser test event: {error}"
            ))
        })?;
        self.0.call1(&JsValue::NULL, &event).map_err(|error| {
            runmat_test_runner::RunnerError::Reporter(format!(
                "browser test event observer failed: {error:?}"
            ))
        })?;
        Ok(())
    }
}
