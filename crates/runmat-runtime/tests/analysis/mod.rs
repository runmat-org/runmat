use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, ApiDeviceInfo, GpuTensorHandle,
    HostTensorOwned, HostTensorView, ThreadProviderGuard,
};
use runmat_analysis_core::{AnalysisFieldValues, AnalysisModel, ReferenceFrame};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_create_model_op, analysis_results_by_run_id_op, analysis_results_op,
    analysis_run_linear_static_with_options, analysis_run_modal_with_options_op,
    analysis_run_nonlinear_with_options_op, analysis_run_transient_with_options_op,
    analysis_validate, AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile,
    AnalysisModalRunOptions, AnalysisNonlinearRunOptions, AnalysisResultsQuery, AnalysisRunOptions,
    AnalysisTransientRunOptions, PrecisionMode, PreconditionerMode, QualityPolicy,
};
use runmat_runtime::geometry::geometry_load_op;
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
enum AnalysisRunKind {
    LinearStatic,
    Modal,
    Transient,
    Nonlinear,
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
    run_kind: AnalysisRunKind,
    expect_validate_error: Option<&'static str>,
    expect_run_error: Option<&'static str>,
    expected_publishable: Option<bool>,
    parity_tolerance: Option<ParityTolerance>,
    gpu_mode: Option<GpuMode>,
    residency_expectation: Option<ResidencyExpectation>,
    max_solver_host_sync_count: Option<u32>,
    min_solver_device_apply_k_ratio: Option<f64>,
    expected_solver_backend: Option<&'static str>,
    modal_mode_count: Option<usize>,
    transient_step_count: Option<usize>,
    max_modal_orthogonality_offdiag: Option<f64>,
    min_modal_relative_frequency_separation: Option<f64>,
    max_transient_residual_norm: Option<f64>,
    max_transient_energy_growth_ratio: Option<f64>,
    min_gpu_speedup_ratio: Option<f64>,
    min_transient_cache_hit_ratio: Option<f64>,
    max_transient_cache_misses: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureManifestEntry {
    id: String,
    description: String,
    run_kind: String,
    expect_validate_error: Option<String>,
    expect_run_error: Option<String>,
    expected_publishable: Option<bool>,
    parity_tolerance_abs: Option<f64>,
    parity_tolerance_rel: Option<f64>,
    gpu_mode: Option<String>,
    residency_expectation: Option<String>,
    max_solver_host_sync_count: Option<u32>,
    min_solver_device_apply_k_ratio: Option<f64>,
    expected_solver_backend: Option<String>,
    modal_mode_count: Option<usize>,
    transient_step_count: Option<usize>,
    max_modal_orthogonality_offdiag: Option<f64>,
    min_modal_relative_frequency_separation: Option<f64>,
    max_transient_residual_norm: Option<f64>,
    max_transient_energy_growth_ratio: Option<f64>,
    min_gpu_speedup_ratio: Option<f64>,
    min_transient_cache_hit_ratio: Option<f64>,
    max_transient_cache_misses: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThresholdAssertionRecord {
    name: String,
    source_diagnostic: String,
    observed: Option<f64>,
    min_allowed: Option<f64>,
    max_allowed: Option<f64>,
    passed: bool,
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
    gpu_solver_host_sync_count: Option<u32>,
    gpu_solver_device_apply_k_ratio: Option<f64>,
    gpu_speedup_ratio: Option<f64>,
    gpu_solver_backend: Option<String>,
    gpu_transient_cache_hit_ratio: Option<f64>,
    gpu_transient_cache_misses: Option<f64>,
    gpu_transient_cache_entries: Option<f64>,
    gpu_solver_prepared_build_ms: Option<f64>,
    gpu_solver_solve_ms: Option<f64>,
    gpu_solver_fallback_apply_count: Option<f64>,
    prep_calibration_profile: Option<String>,
    prep_calibration_fingerprint: Option<u64>,
    prep_acceptance_score: Option<f64>,
    prep_acceptance_passed: Option<bool>,
    prep_acceptance_fingerprint: Option<u64>,
    thermo_coupling_enabled: Option<bool>,
    thermo_coupling_fingerprint: Option<u64>,
    thermo_constitutive_temperature_factor: Option<f64>,
    thermo_effective_modulus_scale: Option<f64>,
    thermo_constitutive_material_spread_ratio: Option<f64>,
    thermo_assignment_heterogeneity_index: Option<f64>,
    thermo_region_delta_count: Option<f64>,
    thermo_spatial_coverage_ratio: Option<f64>,
    thermo_field_extrapolation_ratio: Option<f64>,
    thermo_field_artifact_id: Option<String>,
    thermo_field_artifact_approved: Option<bool>,
    thermo_field_artifact_age_days: Option<f64>,
    thermo_field_artifact_provenance_valid: Option<bool>,
    thermo_transient_severity: Option<f64>,
    thermo_nonlinear_severity: Option<f64>,
    electro_thermal_coupling_enabled: Option<bool>,
    electro_thermal_coupling_fingerprint: Option<u64>,
    electro_joule_heating_scale: Option<f64>,
    electro_conductivity_spread_ratio: Option<f64>,
    electro_transient_severity: Option<f64>,
    electro_nonlinear_severity: Option<f64>,
    plastic_nonlinear_severity: Option<f64>,
    contact_nonlinear_severity: Option<f64>,
    publishable: Option<bool>,
    parity: Option<ParitySummary>,
    threshold_assertions: Vec<ThresholdAssertionRecord>,
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
    rolling_dir: Option<PathBuf>,
    rolling_window: usize,
    enforce: bool,
    max_slowdown_ratio: f64,
    max_cost_slowdown_ratio: f64,
    min_speedup_retention: f64,
}

const ROLLING_TARGET_FIXTURES: &[&str] = &[
    "modal_large_gpu_provider",
    "modal_large_gpu_provider_stress16",
    "transient_long_gpu_provider",
    "transient_shock_gpu_provider",
    "thermo_mech_kickoff_gpu_provider",
    "thermo_gradient_benign_gpu_provider",
    "thermo_gradient_pathological_gpu_provider",
    "thermo_ramp_smooth_gpu_provider",
    "thermo_ramp_smooth_field_artifact_gpu_provider",
    "thermo_shock_oscillatory_gpu_provider",
    "thermo_shock_oscillatory_field_artifact_gpu_provider",
    "electro_thermal_joule_benign_gpu_provider",
    "electro_thermal_joule_pathological_gpu_provider",
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_proxy_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
    "nonlinear_plasticity_proxy_gpu_provider",
    "nonlinear_contact_proxy_gpu_provider",
    "nonlinear_contact_frictionless_reference_gpu_provider",
];

const SYNTHETIC_TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";

fn synthesized_nonlinear_model() -> AnalysisModel {
    let geometry = geometry_load_op(
        "/synthetic/nonlinear_fixture.stl",
        SYNTHETIC_TRIANGLE_STL.as_bytes(),
        OperationContext::new(
            Some("trace-create-model-nonlinear-geometry".to_string()),
            None,
        ),
    )
    .expect("load synthetic nonlinear geometry for create-model fixture");

    let created = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "nonlinear_created_fixture_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-create-model-nonlinear".to_string()), None),
    )
    .expect("synthesize nonlinear model via analysis.create_model");
    created.data
}

mod baseline;
mod harness;
mod manifest;
mod runner;

use baseline::{
    artifact_path, baseline_config, check_baseline_drift, check_rolling_baseline_drift,
    load_baseline_report, load_rolling_reports,
};
use manifest::{fixture_manifest, manifest_specs};
use runner::run_fixture;

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
            failures.push(format!(
                "{} => {}",
                record.fixture_id,
                record.failures.join("; ")
            ));
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
    fs::write(&path, &report_json).expect("write benchmark report artifact");

    if let Ok(dir) = std::env::var("RUNMAT_ANALYSIS_BASELINE_DIR") {
        let dir_path = PathBuf::from(dir);
        if fs::create_dir_all(&dir_path).is_ok() {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_secs())
                .unwrap_or(0);
            let snapshot_path = dir_path.join(format!("analysis_benchmark_report_{stamp}.json"));
            let _ = fs::write(snapshot_path, &report_json);
        }
    }

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
    if let Some(rolling_dir) = baseline.rolling_dir.clone() {
        match load_rolling_reports(&rolling_dir, baseline.rolling_window, &path) {
            Ok(history) => {
                check_rolling_baseline_drift(
                    &history,
                    &report,
                    baseline.max_slowdown_ratio,
                    baseline.max_cost_slowdown_ratio,
                    baseline.min_speedup_retention,
                    &mut failures,
                );
            }
            Err(err) => {
                if baseline.enforce {
                    failures.push(format!(
                        "rolling baseline load failed under enforce mode: {err}"
                    ));
                }
            }
        }
    }

    runmat_runtime::analysis::storage::configure_artifact_store(
        runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::InMemory,
    )
    .expect("reset artifact store to in-memory after harness");

    let core_passed = core_failure_count == 0;
    let baseline_passed = if baseline.enforce {
        failures.is_empty()
    } else {
        true
    };
    assert!(
        core_passed && baseline_passed,
        "analysis benchmark/conformance gates failed: {}",
        failures.join(" | ")
    );
}
