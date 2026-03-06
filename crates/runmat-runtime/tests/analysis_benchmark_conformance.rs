use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_analysis_core::{AnalysisFieldValues, AnalysisModel, ReferenceFrame};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_results_by_run_id_op, analysis_results_op, analysis_run_linear_static_with_options,
    analysis_validate,
    AnalysisResultsQuery, AnalysisRunOptions, PrecisionMode, PreconditionerMode,
};
use runmat_runtime::operations::OperationContext;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
enum GpuMode {
    WithProvider,
    WithoutProvider,
}

#[derive(Debug, Clone, Copy)]
enum ResidencyExpectation {
    DeviceRef,
    HostFallback,
}

#[derive(Debug, Clone, Copy)]
struct ParityTolerance {
    abs: f64,
    rel: f64,
}

#[derive(Debug, Clone)]
struct FixtureSpec {
    id: &'static str,
    description: &'static str,
    model: fn() -> AnalysisModel,
    expect_validate_error: Option<&'static str>,
    expect_run_error: Option<&'static str>,
    expected_publishable: Option<bool>,
    parity_tolerance: Option<ParityTolerance>,
    gpu_mode: Option<GpuMode>,
    residency_expectation: Option<ResidencyExpectation>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureManifestEntry {
    id: String,
    description: String,
    expect_validate_error: Option<String>,
    expect_run_error: Option<String>,
    expected_publishable: Option<bool>,
    parity_tolerance_abs: Option<f64>,
    parity_tolerance_rel: Option<f64>,
    gpu_mode: Option<String>,
    residency_expectation: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureManifest {
    version: String,
    fixtures: Vec<FixtureManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParitySummary {
    max_abs_diff: f64,
    max_rel_diff: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureRunRecord {
    fixture_id: String,
    validate_ok: bool,
    validate_error_code: Option<String>,
    run_ok: bool,
    run_error_code: Option<String>,
    cpu_run_ms: Option<f64>,
    gpu_run_ms: Option<f64>,
    gpu_fallback_events: Vec<String>,
    gpu_displacement_residency: Option<String>,
    publishable: Option<bool>,
    parity: Option<ParitySummary>,
    failures: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkConformanceReport {
    schema_version: String,
    manifest: FixtureManifest,
    records: Vec<FixtureRunRecord>,
    passed: bool,
}

#[derive(Debug)]
struct BaselineConfig {
    path: Option<PathBuf>,
    enforce: bool,
    max_slowdown_ratio: f64,
}

fn manifest_specs() -> Vec<FixtureSpec> {
    vec![
        FixtureSpec {
            id: "cantilever_gpu_provider",
            description: "canonical linear static fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::CantileverLinearStatic),
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
        },
        FixtureSpec {
            id: "cantilever_gpu_fallback",
            description: "canonical linear static fixture with no provider fallback",
            model: || fixture_model(FixtureId::CantileverLinearStatic),
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: Some(ParityTolerance {
                abs: 1e-12,
                rel: 1e-12,
            }),
            gpu_mode: Some(GpuMode::WithoutProvider),
            residency_expectation: Some(ResidencyExpectation::HostFallback),
        },
        FixtureSpec {
            id: "cantilever_load_sweep_gpu_provider",
            description: "larger load sweep fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::CantileverLoadSweep),
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
        },
        FixtureSpec {
            id: "missing_materials",
            description: "invalid fixture must fail validation and run with typed errors",
            model: || fixture_model(FixtureId::MissingMaterials),
            expect_validate_error: Some("ANALYSIS_VALIDATION_MISSING_MATERIALS"),
            expect_run_error: Some("SOLVER_MODEL_INVALID"),
            expected_publishable: None,
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
        },
        FixtureSpec {
            id: "unit_mismatch",
            description: "unit mismatch fixture must fail validation with mismatch code",
            model: || {
                let mut model = fixture_model(FixtureId::CantileverLinearStatic);
                model.units = UnitSystem::Inch;
                model
            },
            expect_validate_error: Some("ANALYSIS_VALIDATION_UNIT_MISMATCH"),
            expect_run_error: None,
            expected_publishable: None,
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
        },
    ]
}

fn fixture_manifest(specs: &[FixtureSpec]) -> FixtureManifest {
    FixtureManifest {
        version: "analysis-conformance/v1".to_string(),
        fixtures: specs
            .iter()
            .map(|spec| FixtureManifestEntry {
                id: spec.id.to_string(),
                description: spec.description.to_string(),
                expect_validate_error: spec.expect_validate_error.map(|s| s.to_string()),
                expect_run_error: spec.expect_run_error.map(|s| s.to_string()),
                expected_publishable: spec.expected_publishable,
                parity_tolerance_abs: spec.parity_tolerance.map(|tol| tol.abs),
                parity_tolerance_rel: spec.parity_tolerance.map(|tol| tol.rel),
                gpu_mode: spec.gpu_mode.map(|mode| match mode {
                    GpuMode::WithProvider => "with_provider".to_string(),
                    GpuMode::WithoutProvider => "without_provider".to_string(),
                }),
                residency_expectation: spec.residency_expectation.map(|value| match value {
                    ResidencyExpectation::DeviceRef => "device_ref".to_string(),
                    ResidencyExpectation::HostFallback => "host_fallback".to_string(),
                }),
            })
            .collect(),
    }
}

fn default_options() -> AnalysisRunOptions {
    AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: PreconditionerMode::Auto,
    }
}

fn validate_fallback_event_schema(event: &str) -> bool {
    let parts: Vec<&str> = event.splitn(3, ':').collect();
    if parts.len() != 3 {
        return false;
    }
    let category_ok = matches!(parts[0], "BACKEND_NO_PROVIDER" | "BACKEND_UPLOAD_FAILED");
    let stage_ok = matches!(parts[1], "displacement" | "von_mises");
    let reason_ok = !parts[2].is_empty();
    category_ok && stage_ok && reason_ok
}

fn compute_parity(left: &[f64], right: &[f64]) -> ParitySummary {
    let mut max_abs = 0.0;
    let mut max_rel = 0.0;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        let abs = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = abs / scale;
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    ParitySummary {
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
    }
}

fn run_fixture(spec: &FixtureSpec, filesystem_root: Option<&PathBuf>) -> FixtureRunRecord {
    let model = (spec.model)();
    let mut failures = Vec::new();

    let validate_start = Instant::now();
    let validate_result = analysis_validate(
        &model,
        UnitSystem::Meter,
        &ReferenceFrame::Global,
        OperationContext::new(Some(format!("trace-validate-{}", spec.id)), None),
    );
    let _validate_ms = validate_start.elapsed().as_secs_f64() * 1_000.0;

    let mut validate_ok = false;
    let mut validate_error_code = None;

    match validate_result {
        Ok(_) => {
            validate_ok = true;
            if let Some(expected) = spec.expect_validate_error {
                failures.push(format!(
                    "expected validate error code {expected}, but validate succeeded"
                ));
            }
        }
        Err(err) => {
            validate_error_code = Some(err.error_code.clone());
            if let Some(expected) = spec.expect_validate_error {
                if err.error_code != expected {
                    failures.push(format!(
                        "validate error mismatch: expected {expected}, got {}",
                        err.error_code
                    ));
                }
            } else {
                failures.push(format!(
                    "unexpected validate error code {} for fixture {}",
                    err.error_code, spec.id
                ));
            }
        }
    }

    let mut run_ok = false;
    let mut run_error_code = None;
    let mut cpu_run_ms = None;
    let mut gpu_run_ms = None;
    let mut gpu_fallback_events = Vec::new();
    let mut gpu_displacement_residency = None;
    let mut publishable = None;
    let mut parity = None;

    if spec.expect_validate_error.is_none() {
        let cpu_start = Instant::now();
        let cpu_result = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            default_options(),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        );
        cpu_run_ms = Some(cpu_start.elapsed().as_secs_f64() * 1_000.0);

        let cpu_envelope = match cpu_result {
            Ok(value) => value,
            Err(err) => {
                failures.push(format!(
                    "unexpected CPU run failure for fixture {}: {}",
                    spec.id, err.error_code
                ));
                return FixtureRunRecord {
                    fixture_id: spec.id.to_string(),
                    validate_ok,
                    validate_error_code,
                    run_ok,
                    run_error_code,
                    cpu_run_ms,
                    gpu_run_ms,
                    gpu_fallback_events,
                    gpu_displacement_residency,
                    publishable,
                    parity,
                    failures,
                };
            }
        };

        if let Some(expected_publishable) = spec.expected_publishable {
            if cpu_envelope.data.publishable != expected_publishable {
                failures.push(format!(
                    "cpu publishable mismatch: expected {expected_publishable}, got {}",
                    cpu_envelope.data.publishable
                ));
            }
        }

        if let Some(gpu_mode) = spec.gpu_mode {
            let gpu_start = Instant::now();
            let gpu_result = match gpu_mode {
                GpuMode::WithProvider => with_harness_provider(|| {
                    analysis_run_linear_static_with_options(
                        &model,
                        ComputeBackend::Gpu,
                        default_options(),
                        OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
                    )
                }),
                GpuMode::WithoutProvider => {
                    let _guard = ThreadProviderGuard::set(None);
                    analysis_run_linear_static_with_options(
                        &model,
                        ComputeBackend::Gpu,
                        default_options(),
                        OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
                    )
                }
            };
            gpu_run_ms = Some(gpu_start.elapsed().as_secs_f64() * 1_000.0);

            match gpu_result {
                Ok(gpu_envelope) => {
                    run_ok = true;
                    publishable = Some(gpu_envelope.data.publishable);
                    gpu_fallback_events = gpu_envelope.data.provenance.fallback_events.clone();

                    for event in &gpu_fallback_events {
                        if !validate_fallback_event_schema(event) {
                            failures.push(format!("invalid fallback event schema: {event}"));
                        }
                    }

                    gpu_displacement_residency = Some(
                        match &gpu_envelope.data.run.displacement_field.values {
                            AnalysisFieldValues::DeviceRef(_) => "device_ref".to_string(),
                            AnalysisFieldValues::HostF64(_) => "host_f64".to_string(),
                        },
                    );

                    if let Some(expected_publishable) = spec.expected_publishable {
                        if gpu_envelope.data.publishable != expected_publishable {
                            failures.push(format!(
                                "gpu publishable mismatch: expected {expected_publishable}, got {}",
                                gpu_envelope.data.publishable
                            ));
                        }
                    }

                    let gpu_results = analysis_results_op(
                        &gpu_envelope.data,
                        AnalysisResultsQuery {
                            include_fields: vec!["displacement".to_string()],
                            include_diagnostics: false,
                        },
                        OperationContext::new(Some(format!("trace-results-gpu-{}", spec.id)), None),
                    );
                    let gpu_results = match gpu_results {
                        Ok(value) => value,
                        Err(err) => {
                            failures.push(format!(
                                "analysis.results failed for gpu fixture {}: {}",
                                spec.id, err.error_code
                            ));
                            return FixtureRunRecord {
                                fixture_id: spec.id.to_string(),
                                validate_ok,
                                validate_error_code,
                                run_ok,
                                run_error_code,
                                cpu_run_ms,
                                gpu_run_ms,
                                gpu_fallback_events,
                                gpu_displacement_residency,
                                publishable,
                                parity,
                                failures,
                            };
                        }
                    };
                    if gpu_results.operation != "analysis.results"
                        || gpu_results.op_version != "analysis.results/v1"
                    {
                        failures.push("analysis.results contract version mismatch".to_string());
                    }

                    if let Some(root) = filesystem_root {
                        runmat_runtime::analysis::storage::configure_artifact_store(
                            runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
                                root: root.clone(),
                            },
                        )
                        .expect("reconfigure filesystem artifact store");

                        let persisted = analysis_results_by_run_id_op(
                            &gpu_envelope.data.run_id,
                            AnalysisResultsQuery {
                                include_fields: vec!["displacement".to_string()],
                                include_diagnostics: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-gpu-by-id-{}", spec.id)),
                                None,
                            ),
                        );
                        match persisted {
                            Ok(by_id) => {
                                if by_id.operation != "analysis.results"
                                    || by_id.op_version != "analysis.results/v1"
                                {
                                    failures.push(
                                        "analysis.results by-run-id contract mismatch".to_string(),
                                    );
                                }
                            }
                            Err(err) => failures.push(format!(
                                "analysis.results by-run-id failed for fixture {}: {}",
                                spec.id, err.error_code
                            )),
                        }
                    }

                    if let Some(expectation) = spec.residency_expectation {
                        match expectation {
                            ResidencyExpectation::DeviceRef => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::DeviceRef(_)
                                ) {
                                    failures.push("expected gpu displacement device_ref residency".to_string());
                                }
                            }
                            ResidencyExpectation::HostFallback => {
                                if !matches!(
                                    &gpu_results.data.fields[0].values,
                                    AnalysisFieldValues::HostF64(_)
                                ) {
                                    failures.push("expected gpu displacement host_f64 fallback".to_string());
                                }
                            }
                        }
                    }

                    if let Some(tol) = spec.parity_tolerance {
                        let cpu_results = analysis_results_op(
                            &cpu_envelope.data,
                            AnalysisResultsQuery {
                                include_fields: vec!["displacement".to_string()],
                                include_diagnostics: false,
                            },
                            OperationContext::new(
                                Some(format!("trace-results-cpu-{}", spec.id)),
                                None,
                            ),
                        );
                        let cpu_results = match cpu_results {
                            Ok(value) => value,
                            Err(err) => {
                                failures.push(format!(
                                    "analysis.results failed for cpu fixture {}: {}",
                                    spec.id, err.error_code
                                ));
                                return FixtureRunRecord {
                                    fixture_id: spec.id.to_string(),
                                    validate_ok,
                                    validate_error_code,
                                    run_ok,
                                    run_error_code,
                                    cpu_run_ms,
                                    gpu_run_ms,
                                    gpu_fallback_events,
                                    gpu_displacement_residency,
                                    publishable,
                                    parity,
                                    failures,
                                };
                            }
                        };

                        let cpu_values = cpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);
                        let gpu_values = gpu_results.data.fields[0].as_host_f64().unwrap_or(&[]);

                        if cpu_values.is_empty() || gpu_values.is_empty() {
                            failures.push(
                                "parity check requested but host vectors were not available".to_string(),
                            );
                        } else if cpu_values.len() != gpu_values.len() {
                            failures.push("parity vector length mismatch".to_string());
                        } else {
                            let summary = compute_parity(cpu_values, gpu_values);
                            if summary.max_abs_diff > tol.abs {
                                failures.push(format!(
                                    "parity abs diff {} exceeds {}",
                                    summary.max_abs_diff, tol.abs
                                ));
                            }
                            if summary.max_rel_diff > tol.rel {
                                failures.push(format!(
                                    "parity rel diff {} exceeds {}",
                                    summary.max_rel_diff, tol.rel
                                ));
                            }
                            parity = Some(summary);
                        }
                    }
                }
                Err(err) => {
                    run_error_code = Some(err.error_code.clone());
                    failures.push(format!(
                        "unexpected gpu run failure for fixture {}: {}",
                        spec.id, err.error_code
                    ));
                }
            }
        }
    }

    if let Some(expected_run_error) = spec.expect_run_error {
        let result = analysis_run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            default_options(),
            OperationContext::new(Some(format!("trace-run-error-{}", spec.id)), None),
        );
        match result {
            Ok(_) => failures.push(format!(
                "expected run error code {expected_run_error}, but run succeeded"
            )),
            Err(err) => {
                run_error_code = Some(err.error_code.clone());
                if err.error_code != expected_run_error {
                    failures.push(format!(
                        "run error mismatch: expected {expected_run_error}, got {}",
                        err.error_code
                    ));
                }
            }
        }
    }

    FixtureRunRecord {
        fixture_id: spec.id.to_string(),
        validate_ok,
        validate_error_code,
        run_ok,
        run_error_code,
        cpu_run_ms,
        gpu_run_ms,
        gpu_fallback_events,
        gpu_displacement_residency,
        publishable,
        parity,
        failures,
    }
}

struct HarnessProvider;

impl AccelProvider for HarnessProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(5000);
        Ok(GpuTensorHandle {
            shape: host.shape.to_vec(),
            device_id: 21,
            buffer_id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
        })
    }

    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostTensorOwned {
                data: vec![0.0; h.shape.iter().product()],
                shape: h.shape.clone(),
            })
        })
    }

    fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn device_info(&self) -> String {
        "analysis-harness-provider".to_string()
    }

    fn device_id(&self) -> u32 {
        21
    }

    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 21,
            name: "analysis-harness-provider".to_string(),
            vendor: "runmat-tests".to_string(),
            memory_bytes: None,
            backend: Some("harness_gpu".to_string()),
        }
    }
}

fn with_harness_provider<T>(f: impl FnOnce() -> T) -> T {
    static PROVIDER: HarnessProvider = HarnessProvider;
    let _guard = ThreadProviderGuard::set(Some(&PROVIDER));
    f()
}

fn artifact_path() -> PathBuf {
    if let Ok(path) = std::env::var("RUNMAT_ANALYSIS_ARTIFACT_PATH") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/runmat-analysis-artifacts/analysis_benchmark_report.json")
}

fn baseline_config() -> BaselineConfig {
    let path = std::env::var("RUNMAT_ANALYSIS_BASELINE_PATH")
        .ok()
        .map(PathBuf::from);
    let enforce = std::env::var("RUNMAT_ANALYSIS_ENFORCE_BASELINE")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let max_slowdown_ratio = std::env::var("RUNMAT_ANALYSIS_MAX_SLOWDOWN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.5);
    BaselineConfig {
        path,
        enforce,
        max_slowdown_ratio,
    }
}

fn load_baseline_report(path: &PathBuf) -> Result<BenchmarkConformanceReport, String> {
    let bytes = fs::read(path).map_err(|err| format!("failed to read baseline report: {err}"))?;
    serde_json::from_slice::<BenchmarkConformanceReport>(&bytes)
        .map_err(|err| format!("failed to parse baseline report: {err}"))
}

fn check_baseline_drift(
    baseline: &BenchmarkConformanceReport,
    current: &BenchmarkConformanceReport,
    max_slowdown_ratio: f64,
    failures: &mut Vec<String>,
) {
    for current_record in &current.records {
        let Some(baseline_record) = baseline
            .records
            .iter()
            .find(|record| record.fixture_id == current_record.fixture_id)
        else {
            continue;
        };

        check_timing_ratio(
            &current_record.fixture_id,
            "cpu",
            baseline_record.cpu_run_ms,
            current_record.cpu_run_ms,
            max_slowdown_ratio,
            failures,
        );
        check_timing_ratio(
            &current_record.fixture_id,
            "gpu",
            baseline_record.gpu_run_ms,
            current_record.gpu_run_ms,
            max_slowdown_ratio,
            failures,
        );
    }
}

fn check_timing_ratio(
    fixture_id: &str,
    backend: &str,
    baseline_ms: Option<f64>,
    current_ms: Option<f64>,
    max_slowdown_ratio: f64,
    failures: &mut Vec<String>,
) {
    let (Some(base), Some(now)) = (baseline_ms, current_ms) else {
        return;
    };
    if base <= 0.0 {
        return;
    }
    let ratio = now / base;
    if ratio > max_slowdown_ratio {
        failures.push(format!(
            "baseline slowdown exceeded for fixture={} backend={} ratio={ratio:.3} limit={max_slowdown_ratio:.3} (baseline_ms={base:.3}, current_ms={now:.3})",
            fixture_id,
            backend
        ));
    }
}

#[test]
fn analysis_benchmark_conformance_manifest_gates() {
    let store_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/runmat-analysis-artifacts/store")
        .join(format!("harness-{}", std::process::id()));
    let _ = fs::remove_dir_all(&store_root);
    runmat_runtime::analysis::storage::configure_artifact_store(
        runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
            root: store_root.clone(),
        },
    )
    .expect("configure filesystem artifact store for harness");

    let specs = manifest_specs();
    let manifest = fixture_manifest(&specs);

    assert_eq!(manifest.version, "analysis-conformance/v1");

    let records: Vec<FixtureRunRecord> = specs
        .iter()
        .map(|spec| run_fixture(spec, Some(&store_root)))
        .collect();
    let mut failures = Vec::new();
    for record in &records {
        if !record.failures.is_empty() {
            failures.push(format!("{} => {}", record.fixture_id, record.failures.join("; ")));
        }
    }

    let report = BenchmarkConformanceReport {
        schema_version: "analysis-benchmark-report/v1".to_string(),
        manifest,
        records,
        passed: failures.is_empty(),
    };

    let report_json = serde_json::to_string_pretty(&report).expect("serialize report");
    let path = artifact_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create artifact directory");
    }
    fs::write(&path, report_json).expect("write benchmark report artifact");

    let baseline = baseline_config();
    let core_failure_count = failures.len();
    if let Some(path) = baseline.path {
        match load_baseline_report(&path) {
            Ok(previous) => {
                check_baseline_drift(
                    &previous,
                    &report,
                    baseline.max_slowdown_ratio,
                    &mut failures,
                );
            }
            Err(err) => {
                if baseline.enforce {
                    failures.push(format!("baseline load failed under enforce mode: {err}"));
                }
            }
        }
    }

    runmat_runtime::analysis::storage::configure_artifact_store(
        runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::InMemory,
    )
    .expect("reset artifact store to in-memory after harness");

    assert!(
        if baseline.enforce {
            failures.is_empty()
        } else {
            failures.len() == core_failure_count
        },
        "analysis benchmark/conformance gates failed: {}",
        failures.join(" | ")
    );
}
