use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_analysis_core::{AnalysisFieldValues, AnalysisModel, ProcedureStep};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_meshing_core::{
    solver_volume_element_identity, CanonicalMeshingContract, ElementOrder, FieldTopologyLocation,
    MeshNeighbor, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, SolverMeshNode,
    SolverVolumeElement, StableDigest,
};
use runmat_runtime::analysis::{
    analysis_results_by_run_id_op, analysis_results_op, analysis_run_acoustic_with_options_op,
    analysis_run_cfd_with_options_op, analysis_run_cht_op, analysis_run_fsi_op,
    analysis_run_linear_static_with_options, analysis_run_modal_with_options_op,
    analysis_run_nonlinear_with_options_op, analysis_run_thermal_with_options_op,
    analysis_run_transient_with_options_op, analysis_validate, AnalysisAcousticRunOptions,
    AnalysisCfdRunOptions, AnalysisModalRunOptions, AnalysisNonlinearRunOptions,
    AnalysisResultsQuery, AnalysisRunOptions, AnalysisSolverOptions,
    AnalysisStructuralTimeStepPolicy, AnalysisThermalRunOptions, AnalysisTransientRunOptions,
    FallbackEvent, FallbackReason, FallbackStage, PrecisionMode, PreconditionerMode, QualityPolicy,
};
use runmat_runtime::operations::OperationContext;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
enum GpuMode {
    WithProvider,
    WithoutProvider,
}

#[derive(Debug, Clone, Copy)]
enum PublishedFieldStorageExpectation {
    HostF64,
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
    /// The GPU placement's rejection when it differs from the CPU placement's. Static GPU
    /// execution fails closed at backend admission before the model reaches assembly.
    expected_numerically_acceptable: Option<bool>,
    parity_tolerance: Option<ParityTolerance>,
    gpu_mode: Option<GpuMode>,
    published_field_storage_expectation: Option<PublishedFieldStorageExpectation>,
    max_solver_host_sync_count: Option<u32>,
    min_solver_device_apply_k_ratio: Option<f64>,
    expected_solver_backend: Option<&'static str>,
    modal_mode_count: Option<usize>,
    transient_step_count: Option<usize>,
    max_modal_orthogonality_offdiag: Option<f64>,
    min_modal_relative_frequency_separation: Option<f64>,
    max_transient_residual_norm: Option<f64>,
    min_gpu_speedup_ratio: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FixtureManifestEntry {
    id: String,
    description: String,
    procedure_step: ProcedureStep,
    expect_validate_error: Option<String>,
    expect_run_error: Option<String>,
    expected_numerically_acceptable: Option<bool>,
    parity_tolerance_abs: Option<f64>,
    parity_tolerance_rel: Option<f64>,
    gpu_mode: Option<String>,
    published_field_storage_expectation: Option<String>,
    max_solver_host_sync_count: Option<u32>,
    min_solver_device_apply_k_ratio: Option<f64>,
    expected_solver_backend: Option<String>,
    modal_mode_count: Option<usize>,
    transient_step_count: Option<usize>,
    max_modal_orthogonality_offdiag: Option<f64>,
    min_modal_relative_frequency_separation: Option<f64>,
    max_transient_residual_norm: Option<f64>,
    min_gpu_speedup_ratio: Option<f64>,
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

fn fixture_procedure_step(model: &AnalysisModel) -> &ProcedureStep {
    let [step] = model.sequence.steps.as_slice() else {
        panic!(
            "analysis fixture `{}` must define exactly one canonical procedure step; found {}",
            model.model_id.0,
            model.sequence.steps.len()
        )
    };
    step
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
    gpu_fallback_events: Vec<FallbackEvent>,
    published_primary_field_storage: Option<String>,
    gpu_solver_host_sync_count: Option<u32>,
    gpu_solver_device_apply_k_ratio: Option<f64>,
    gpu_speedup_ratio: Option<f64>,
    gpu_solver_backend: Option<String>,
    gpu_solver_prepared_build_ms: Option<f64>,
    gpu_solver_solve_ms: Option<f64>,
    gpu_solver_fallback_apply_count: Option<f64>,
    thermo_coupling_enabled: Option<bool>,
    thermo_coupling_fingerprint: Option<u64>,
    thermo_constitutive_temperature_factor: Option<f64>,
    thermo_effective_modulus_scale: Option<f64>,
    thermo_constitutive_material_spread_ratio: Option<f64>,
    thermo_assignment_heterogeneity_index: Option<f64>,
    thermo_region_delta_count: Option<f64>,
    thermo_spatial_coverage_ratio: Option<f64>,
    thermo_nonlinear_severity: Option<f64>,
    electro_thermal_coupling_enabled: Option<bool>,
    electro_thermal_coupling_fingerprint: Option<u64>,
    electro_joule_heating_scale: Option<f64>,
    electro_conductivity_spread_ratio: Option<f64>,
    electro_transient_severity: Option<f64>,
    electro_nonlinear_severity: Option<f64>,
    plastic_nonlinear_severity: Option<f64>,
    plastic_nonlinear_severity_mean: Option<f64>,
    plastic_load_realization_ratio: Option<f64>,
    plastic_load_amplification_ratio: Option<f64>,
    contact_nonlinear_severity: Option<f64>,
    contact_nonlinear_severity_mean: Option<f64>,
    contact_load_realization_ratio: Option<f64>,
    contact_load_amplification_ratio: Option<f64>,
    thermal_max_relative_residual_norm: Option<f64>,
    thermal_min_temperature_k: Option<f64>,
    thermal_max_temperature_k: Option<f64>,
    thermal_conductivity_spread_ratio: Option<f64>,
    thermal_heat_capacity_spread_ratio: Option<f64>,
    electromagnetic_enabled: Option<bool>,
    electromagnetic_formulation_coverage_ratio: Option<f64>,
    electromagnetic_magnetostatic_curl_curl_coverage_ratio: Option<f64>,
    electromagnetic_magnetoquasistatic_eddy_current_coverage_ratio: Option<f64>,
    electromagnetic_full_wave_displacement_current_coverage_ratio: Option<f64>,
    electromagnetic_displacement_to_conduction_ratio: Option<f64>,
    electromagnetic_material_frequency_response_coverage_ratio: Option<f64>,
    electromagnetic_reference_frequency_hz: Option<f64>,
    electromagnetic_applied_current_a: Option<f64>,
    electromagnetic_solve_quality: Option<f64>,
    electromagnetic_conductivity_spread_ratio: Option<f64>,
    electromagnetic_relative_permittivity_spread_ratio: Option<f64>,
    electromagnetic_relative_permeability_spread_ratio: Option<f64>,
    electromagnetic_material_heterogeneity_index: Option<f64>,
    electromagnetic_assignment_coverage_ratio: Option<f64>,
    electromagnetic_assigned_coefficient_coverage_ratio: Option<f64>,
    electromagnetic_region_coefficient_contrast_index: Option<f64>,
    electromagnetic_condition_number_estimate: Option<f64>,
    electromagnetic_source_realization_ratio: Option<f64>,
    electromagnetic_source_region_coverage_ratio: Option<f64>,
    electromagnetic_source_material_alignment_ratio: Option<f64>,
    electromagnetic_source_localization_ratio: Option<f64>,
    electromagnetic_source_overlap_ratio: Option<f64>,
    electromagnetic_source_interference_index: Option<f64>,
    electromagnetic_boundary_anchor_ratio: Option<f64>,
    electromagnetic_boundary_condition_localization_ratio: Option<f64>,
    electromagnetic_ground_anchor_effectiveness_ratio: Option<f64>,
    electromagnetic_insulation_leakage_ratio: Option<f64>,
    electromagnetic_flux_divergence_ratio: Option<f64>,
    electromagnetic_energy_imbalance_ratio: Option<f64>,
    electromagnetic_boundary_energy_ratio: Option<f64>,
    electromagnetic_boundary_penalty_conditioning_contribution: Option<f64>,
    electromagnetic_source_region_energy_consistency_ratio: Option<f64>,
    electromagnetic_real_residual_norm: Option<f64>,
    electromagnetic_imag_residual_norm: Option<f64>,
    electromagnetic_sweep_count: Option<f64>,
    electromagnetic_resonance_peak_frequency_hz: Option<f64>,
    electromagnetic_resonance_peak_flux_density: Option<f64>,
    electromagnetic_resonance_bandwidth_hz: Option<f64>,
    electromagnetic_resonance_quality_factor: Option<f64>,
    electromagnetic_resonance_flux_gain: Option<f64>,
    numerically_acceptable: Option<bool>,
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
    "cht_coupled_gpu_provider",
    "fsi_coupled_gpu_provider",
    "thermal_standalone_ramp_gpu_provider",
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_benchmark_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
    "nonlinear_plasticity_benchmark_gpu_provider",
    "nonlinear_contact_benchmark_gpu_provider",
    "nonlinear_contact_frictionless_reference_gpu_provider",
    "nonlinear_contact_frictionless_reference_complex_gpu_provider",
    "nonlinear_plastic_hardening_reference_gpu_provider",
    "nonlinear_plastic_hardening_reference_complex_gpu_provider",
    "electromagnetic_reference_homogeneous_gpu_provider",
    "electromagnetic_reference_heterogeneous_gpu_provider",
    "electromagnetic_reference_sparse_assignments_gpu_provider",
    "electromagnetic_reference_multiregion_assignments_gpu_provider",
    "electromagnetic_reference_overlap_interference_gpu_provider",
    "electromagnetic_reference_boundary_kernel_gpu_provider",
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider",
    "electromagnetic_reference_multi_region_phased_source_gpu_provider",
];

fn default_harness_artifact_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/runmat-fea-artifacts")
}

fn env_path(primary: &str, legacy: &str) -> Option<PathBuf> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
        .map(PathBuf::from)
}

fn harness_artifact_root() -> PathBuf {
    env_path("RUNMAT_FEA_ARTIFACT_ROOT", "RUNMAT_ANALYSIS_ARTIFACT_ROOT")
        .unwrap_or_else(default_harness_artifact_root)
}

fn synthesized_nonlinear_model() -> AnalysisModel {
    fixture_model(FixtureId::NonlinearAssembly)
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
fn fea_conformance_manifest_uses_physical_fixture_language() {
    let forbidden_terms = ["proxy", "placeholder", "baseline"];
    for spec in manifest_specs() {
        let id = spec.id.to_ascii_lowercase();
        let description = spec.description.to_ascii_lowercase();
        for term in forbidden_terms {
            assert!(
                !id.contains(term),
                "fixture id '{}' contains forbidden interim term '{}'",
                spec.id,
                term
            );
            assert!(
                !description.contains(term),
                "fixture description for '{}' contains forbidden interim term '{}'",
                spec.id,
                term
            );
        }
    }
}

#[test]
fn fea_prescribed_motion_release_fixture_gates() {
    let _admission = crate::fea_admission::guard();
    let spec = manifest_specs()
        .into_iter()
        .find(|spec| spec.id == "transient_physical_tet4_support_motion_reference_cpu")
        .expect("the prescribed-motion release fixture must remain in the manifest");
    let root = tempfile::TempDir::new().expect("temporary release-fixture artifact root");
    let root = root.path().to_path_buf();
    let record = run_fixture(&spec, Some(&root));
    assert!(
        record.failures.is_empty(),
        "prescribed-motion release fixture failed: {}",
        record.failures.join("; ")
    );
}

#[test]
fn fea_benchmark_conformance_manifest_gates() {
    let _admission = crate::fea_admission::guard();
    let artifact_root = harness_artifact_root();
    let store_root = artifact_root
        .join("store")
        .join(format!("harness-{}", std::process::id()));
    let _ = fs::remove_dir_all(&store_root);
    runmat_runtime::analysis::storage::configure_artifact_store(
        runmat_runtime::analysis::storage::AnalysisArtifactStoreConfig::Filesystem {
            root: store_root.clone(),
        },
    )
    .expect("configure filesystem artifact store for harness");
    runmat_runtime::analysis::configure_fea_runtime(runmat_runtime::analysis::FeaRuntimeConfig {
        artifact_root: Some(artifact_root.clone()),
        study_artifact_root: Some(artifact_root.join("studies")),
        ..Default::default()
    })
    .expect("configure FEA runtime artifact roots for harness");

    let specs = manifest_specs();
    let manifest = fixture_manifest(&specs);

    assert_eq!(manifest.version, "fea-conformance/v1");

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
        schema_version: "fea-benchmark-report/v1".to_string(),
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

    if let Some(dir) = std::env::var("RUNMAT_FEA_BASELINE_DIR")
        .ok()
        .or_else(|| std::env::var("RUNMAT_ANALYSIS_BASELINE_DIR").ok())
    {
        let dir_path = PathBuf::from(dir);
        if fs::create_dir_all(&dir_path).is_ok() {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_secs())
                .unwrap_or(0);
            let report_path =
                dir_path.join(format!("analysis_benchmark_report_rolling_{stamp}.json"));
            let _ = fs::write(report_path, &report_json);
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
        "FEA benchmark/conformance gates failed: {}",
        failures.join(" | ")
    );
}
