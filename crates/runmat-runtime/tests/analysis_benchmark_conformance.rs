use std::fs;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView, ThreadProviderGuard,
};
use runmat_analysis_core::{AnalysisFieldValues, AnalysisModel, ReferenceFrame};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::UnitSystem;
use runmat_runtime::analysis::{
    analysis_results_by_run_id_op, analysis_results_op, analysis_run_linear_static_with_options,
    analysis_run_modal_with_options_op, analysis_run_transient_with_options_op, analysis_validate,
    AnalysisModalRunOptions, AnalysisResultsQuery, AnalysisRunOptions, AnalysisTransientRunOptions,
    PrecisionMode, PreconditionerMode, QualityPolicy,
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
enum AnalysisRunKind {
    LinearStatic,
    Modal,
    Transient,
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

const ROLLING_TARGET_FIXTURES: &[&str] =
    &[
        "modal_large_gpu_provider",
        "modal_large_gpu_provider_stress16",
        "transient_long_gpu_provider",
        "transient_shock_gpu_provider",
    ];

fn manifest_specs() -> Vec<FixtureSpec> {
    vec![
        FixtureSpec {
            id: "cantilever_gpu_provider",
            description: "canonical linear static fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::CantileverLinearStatic),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(32),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "cantilever_gpu_fallback",
            description: "canonical linear static fixture with no provider fallback",
            model: || fixture_model(FixtureId::CantileverLinearStatic),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: Some(ParityTolerance {
                abs: 1e-12,
                rel: 1e-12,
            }),
            gpu_mode: Some(GpuMode::WithoutProvider),
            residency_expectation: Some(ResidencyExpectation::HostFallback),
            max_solver_host_sync_count: Some(0),
            min_solver_device_apply_k_ratio: Some(0.0),
            expected_solver_backend: Some("cpu_reference"),
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "cantilever_load_sweep_gpu_provider",
            description: "larger load sweep fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::CantileverLoadSweep),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(32),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "cantilever_large_load_sweep_gpu_provider",
            description: "extra-large load sweep fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::CantileverLargeLoadSweep),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(64),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "multi_material_assembly_gpu_provider",
            description: "multi-material, multi-load assembly fixture with provider-backed GPU residency",
            model: || fixture_model(FixtureId::MultiMaterialAssembly),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(48),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "modal_large_cpu",
            description: "large modal fixture with thresholded orthogonality/separation diagnostics",
            model: || fixture_model(FixtureId::ModalLarge),
            run_kind: AnalysisRunKind::Modal,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: Some(8),
            transient_step_count: None,
            max_modal_orthogonality_offdiag: Some(5.0e-3),
            min_modal_relative_frequency_separation: Some(1.0e-5),
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "modal_large_gpu_provider",
            description: "large modal fixture with provider-backed field residency",
            model: || fixture_model(FixtureId::ModalLarge),
            run_kind: AnalysisRunKind::Modal,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(96),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: Some(8),
            transient_step_count: None,
            max_modal_orthogonality_offdiag: Some(5.0e-3),
            min_modal_relative_frequency_separation: Some(1.0e-5),
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: Some(2.8),
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "modal_large_gpu_fallback",
            description: "large modal fixture with no-provider host fallback",
            model: || fixture_model(FixtureId::ModalLarge),
            run_kind: AnalysisRunKind::Modal,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: Some(ParityTolerance {
                abs: 1e-12,
                rel: 1e-12,
            }),
            gpu_mode: Some(GpuMode::WithoutProvider),
            residency_expectation: Some(ResidencyExpectation::HostFallback),
            max_solver_host_sync_count: Some(0),
            min_solver_device_apply_k_ratio: Some(0.0),
            expected_solver_backend: Some("cpu_reference"),
            modal_mode_count: Some(8),
            transient_step_count: None,
            max_modal_orthogonality_offdiag: Some(5.0e-3),
            min_modal_relative_frequency_separation: Some(1.0e-5),
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "modal_large_cpu_stress16",
            description: "large modal fixture stress run at 16 modes",
            model: || fixture_model(FixtureId::ModalLarge),
            run_kind: AnalysisRunKind::Modal,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: Some(16),
            transient_step_count: None,
            max_modal_orthogonality_offdiag: Some(8.0e-3),
            min_modal_relative_frequency_separation: Some(5.0e-6),
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "modal_large_gpu_provider_stress16",
            description: "large modal fixture GPU provider stress run at 16 modes",
            model: || fixture_model(FixtureId::ModalLarge),
            run_kind: AnalysisRunKind::Modal,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(false),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(144),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: Some(16),
            transient_step_count: None,
            max_modal_orthogonality_offdiag: Some(8.0e-3),
            min_modal_relative_frequency_separation: Some(5.0e-6),
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: Some(2.2),
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "transient_long_cpu",
            description: "long transient fixture with thresholded stability diagnostics",
            model: || fixture_model(FixtureId::TransientLong),
            run_kind: AnalysisRunKind::Transient,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: None,
            transient_step_count: Some(24),
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: Some(1.0e-2),
            max_transient_energy_growth_ratio: Some(5.0),
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "transient_long_gpu_provider",
            description: "long transient fixture with provider-backed field residency",
            model: || fixture_model(FixtureId::TransientLong),
            run_kind: AnalysisRunKind::Transient,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(48),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: Some(24),
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: Some(1.0e-2),
            max_transient_energy_growth_ratio: Some(5.0),
            min_gpu_speedup_ratio: Some(2.0),
            min_transient_cache_hit_ratio: Some(0.2),
            max_transient_cache_misses: Some(24.0),
        },
        FixtureSpec {
            id: "transient_long_gpu_fallback",
            description: "long transient fixture with no-provider host fallback",
            model: || fixture_model(FixtureId::TransientLong),
            run_kind: AnalysisRunKind::Transient,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: Some(ParityTolerance {
                abs: 1e-12,
                rel: 1e-12,
            }),
            gpu_mode: Some(GpuMode::WithoutProvider),
            residency_expectation: Some(ResidencyExpectation::HostFallback),
            max_solver_host_sync_count: Some(0),
            min_solver_device_apply_k_ratio: Some(0.0),
            expected_solver_backend: Some("cpu_reference"),
            modal_mode_count: None,
            transient_step_count: Some(24),
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: Some(1.0e-2),
            max_transient_energy_growth_ratio: Some(5.0),
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "transient_shock_cpu",
            description: "challenging transient fixture with mixed-load shock profile",
            model: || fixture_model(FixtureId::TransientShock),
            run_kind: AnalysisRunKind::Transient,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: None,
            transient_step_count: Some(48),
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: Some(2.0e-2),
            max_transient_energy_growth_ratio: Some(8.0),
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "transient_shock_gpu_provider",
            description: "challenging transient fixture with provider-backed acceleration",
            model: || fixture_model(FixtureId::TransientShock),
            run_kind: AnalysisRunKind::Transient,
            expect_validate_error: None,
            expect_run_error: None,
            expected_publishable: Some(true),
            parity_tolerance: None,
            gpu_mode: Some(GpuMode::WithProvider),
            residency_expectation: Some(ResidencyExpectation::DeviceRef),
            max_solver_host_sync_count: Some(96),
            min_solver_device_apply_k_ratio: Some(1.0),
            expected_solver_backend: Some("runtime_tensor"),
            modal_mode_count: None,
            transient_step_count: Some(48),
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: Some(2.0e-2),
            max_transient_energy_growth_ratio: Some(8.0),
            min_gpu_speedup_ratio: Some(1.8),
            min_transient_cache_hit_ratio: Some(0.15),
            max_transient_cache_misses: Some(40.0),
        },
        FixtureSpec {
            id: "missing_materials",
            description: "invalid fixture must fail validation and run with typed errors",
            model: || fixture_model(FixtureId::MissingMaterials),
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: Some("ANALYSIS_VALIDATION_MISSING_MATERIALS"),
            expect_run_error: Some("SOLVER_MODEL_INVALID"),
            expected_publishable: None,
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
        },
        FixtureSpec {
            id: "unit_mismatch",
            description: "unit mismatch fixture must fail validation with mismatch code",
            model: || {
                let mut model = fixture_model(FixtureId::CantileverLinearStatic);
                model.units = UnitSystem::Inch;
                model
            },
            run_kind: AnalysisRunKind::LinearStatic,
            expect_validate_error: Some("ANALYSIS_VALIDATION_UNIT_MISMATCH"),
            expect_run_error: None,
            expected_publishable: None,
            parity_tolerance: None,
            gpu_mode: None,
            residency_expectation: None,
            max_solver_host_sync_count: None,
            min_solver_device_apply_k_ratio: None,
            expected_solver_backend: None,
            modal_mode_count: None,
            transient_step_count: None,
            max_modal_orthogonality_offdiag: None,
            min_modal_relative_frequency_separation: None,
            max_transient_residual_norm: None,
            max_transient_energy_growth_ratio: None,
            min_gpu_speedup_ratio: None,
            min_transient_cache_hit_ratio: None,
            max_transient_cache_misses: None,
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
                run_kind: match spec.run_kind {
                    AnalysisRunKind::LinearStatic => "linear_static".to_string(),
                    AnalysisRunKind::Modal => "modal".to_string(),
                    AnalysisRunKind::Transient => "transient".to_string(),
                },
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
                max_solver_host_sync_count: spec.max_solver_host_sync_count,
                min_solver_device_apply_k_ratio: spec.min_solver_device_apply_k_ratio,
                expected_solver_backend: spec.expected_solver_backend.map(|s| s.to_string()),
                modal_mode_count: spec.modal_mode_count,
                transient_step_count: spec.transient_step_count,
                max_modal_orthogonality_offdiag: spec.max_modal_orthogonality_offdiag,
                min_modal_relative_frequency_separation: spec
                    .min_modal_relative_frequency_separation,
                max_transient_residual_norm: spec.max_transient_residual_norm,
                max_transient_energy_growth_ratio: spec.max_transient_energy_growth_ratio,
                min_gpu_speedup_ratio: spec.min_gpu_speedup_ratio,
                min_transient_cache_hit_ratio: spec.min_transient_cache_hit_ratio,
                max_transient_cache_misses: spec.max_transient_cache_misses,
            })
            .collect(),
    }
}

fn default_options() -> AnalysisRunOptions {
    AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: PreconditionerMode::Auto,
        quality_policy: QualityPolicy::Balanced,
    }
}

fn run_fixture_cpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
) -> Result<
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>,
    runmat_runtime::operations::OperationErrorEnvelope,
> {
    match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Cpu,
            default_options(),
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Cpu,
            AnalysisModalRunOptions {
                mode_count: spec.modal_mode_count.unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Cpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                dt_bucket_rel_tolerance: requested_bucket_rel_tol
                    .unwrap_or(AnalysisTransientRunOptions::production_recommended().dt_bucket_rel_tolerance),
                ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-cpu-{}", spec.id)), None),
        ),
    }
}

fn run_fixture_gpu(
    spec: &FixtureSpec,
    model: &AnalysisModel,
    mode: GpuMode,
) -> Result<
    runmat_runtime::operations::OperationEnvelope<runmat_runtime::analysis::AnalysisRunResult>,
    runmat_runtime::operations::OperationErrorEnvelope,
> {
    let run = || match spec.run_kind {
        AnalysisRunKind::LinearStatic => analysis_run_linear_static_with_options(
            model,
            ComputeBackend::Gpu,
            default_options(),
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Modal => analysis_run_modal_with_options_op(
            model,
            ComputeBackend::Gpu,
            AnalysisModalRunOptions {
                mode_count: spec.modal_mode_count.unwrap_or(AnalysisModalRunOptions::default().mode_count),
                ..AnalysisModalRunOptions::balanced()
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
        AnalysisRunKind::Transient => analysis_run_transient_with_options_op(
            model,
            ComputeBackend::Gpu,
            {
                let requested_bucket_rel_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                    .ok()
                    .and_then(|value| value.parse::<f64>().ok());
                AnalysisTransientRunOptions {
                step_count: spec
                    .transient_step_count
                    .unwrap_or(AnalysisTransientRunOptions::default().step_count),
                dt_bucket_rel_tolerance: requested_bucket_rel_tol
                    .unwrap_or(AnalysisTransientRunOptions::production_recommended().dt_bucket_rel_tolerance),
                ..AnalysisTransientRunOptions::production_recommended()
                }
            },
            OperationContext::new(Some(format!("trace-gpu-{}", spec.id)), None),
        ),
    };
    match mode {
        GpuMode::WithProvider => with_harness_provider(run),
        GpuMode::WithoutProvider => {
            let _guard = ThreadProviderGuard::set(None);
            run()
        }
    }
}

fn parse_metric_value(message: &str, key: &str) -> Option<f64> {
    message
        .split_whitespace()
        .find_map(|token| token.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.parse::<f64>().ok())
}

fn diagnostic_metric(
    run: &runmat_runtime::analysis::AnalysisRunResult,
    code: &str,
    key: &str,
) -> Option<f64> {
    run.run
        .diagnostics
        .iter()
        .find(|diag| diag.code == code)
        .and_then(|diag| parse_metric_value(&diag.message, key))
}

fn push_threshold_assertion(
    fixture_id: &str,
    assertions: &mut Vec<ThresholdAssertionRecord>,
    failures: &mut Vec<String>,
    name: &str,
    source_diagnostic: &str,
    observed: Option<f64>,
    min_allowed: Option<f64>,
    max_allowed: Option<f64>,
) {
    let passed = observed
        .map(|value| {
            min_allowed.map(|min| value >= min).unwrap_or(true)
                && max_allowed.map(|max| value <= max).unwrap_or(true)
        })
        .unwrap_or(false);
    assertions.push(ThresholdAssertionRecord {
        name: name.to_string(),
        source_diagnostic: source_diagnostic.to_string(),
        observed,
        min_allowed,
        max_allowed,
        passed,
    });
    if !passed {
        failures.push(format!(
            "threshold assertion failed for fixture {}: {} observed={:?} min={:?} max={:?}",
            fixture_id, name, observed, min_allowed, max_allowed
        ));
    }
}

fn validate_fallback_event_schema(event: &str) -> bool {
    let parts: Vec<&str> = event.splitn(3, ':').collect();
    if parts.len() != 3 {
        return false;
    }
    let category_ok = matches!(
        parts[0],
        "BACKEND_NO_PROVIDER" | "BACKEND_UPLOAD_FAILED" | "SOLVER_BACKEND_FALLBACK"
    );
    let stage_ok = if parts[0] == "SOLVER_BACKEND_FALLBACK" {
        parts[1].starts_with("requested=")
    } else {
        matches!(parts[1], "displacement" | "von_mises")
    };
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
    let mut gpu_solver_host_sync_count = None;
    let mut gpu_solver_device_apply_k_ratio = None;
    let mut gpu_speedup_ratio = None;
    let mut gpu_solver_backend = None;
    let mut gpu_transient_cache_hit_ratio = None;
    let mut gpu_transient_cache_misses = None;
    let mut gpu_transient_cache_entries = None;
    let mut gpu_solver_prepared_build_ms = None;
    let mut gpu_solver_solve_ms = None;
    let mut gpu_solver_fallback_apply_count = None;
    let mut publishable = None;
    let mut parity = None;
    let mut threshold_assertions = Vec::new();

    if spec.expect_validate_error.is_none() {
        let cpu_start = Instant::now();
        let cpu_result = run_fixture_cpu(spec, &model);
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
                    gpu_solver_host_sync_count,
                    gpu_solver_device_apply_k_ratio,
                    gpu_speedup_ratio,
                    gpu_solver_backend,
                    gpu_transient_cache_hit_ratio,
                    gpu_transient_cache_misses,
                    gpu_transient_cache_entries,
                    gpu_solver_prepared_build_ms,
                    gpu_solver_solve_ms,
                    gpu_solver_fallback_apply_count,
                    publishable,
                    parity,
                    threshold_assertions,
                    failures,
                };
            }
        };
        run_ok = true;
        publishable = Some(cpu_envelope.data.publishable);

        if let Some(expected_publishable) = spec.expected_publishable {
            if cpu_envelope.data.publishable != expected_publishable {
                failures.push(format!(
                    "cpu publishable mismatch: expected {expected_publishable}, got {}",
                    cpu_envelope.data.publishable
                ));
            }
        }

        if let Some(max_offdiag) = spec.max_modal_orthogonality_offdiag {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_ORTHOGONALITY",
                "max_m_orthogonality_offdiag",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_max_m_orthogonality_offdiag",
                "FEA_MODAL_ORTHOGONALITY",
                observed,
                None,
                Some(max_offdiag),
            );
        }
        if let Some(min_separation) = spec.min_modal_relative_frequency_separation {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_MODAL_SEPARATION",
                "min_relative_frequency_separation",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "modal_min_relative_frequency_separation",
                "FEA_MODAL_SEPARATION",
                observed,
                Some(min_separation),
                None,
            );
        }
        if let Some(max_residual) = spec.max_transient_residual_norm {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_STABILITY",
                "max_residual_norm",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_residual_norm",
                "FEA_TRANSIENT_STABILITY",
                observed,
                None,
                Some(max_residual),
            );
        }
        if let Some(max_growth) = spec.max_transient_energy_growth_ratio {
            let observed = diagnostic_metric(
                &cpu_envelope.data,
                "FEA_TRANSIENT_ENERGY",
                "max_energy_growth_ratio",
            );
            push_threshold_assertion(
                spec.id,
                &mut threshold_assertions,
                &mut failures,
                "transient_max_energy_growth_ratio",
                "FEA_TRANSIENT_ENERGY",
                observed,
                None,
                Some(max_growth),
            );
        }

        if let Some(gpu_mode) = spec.gpu_mode {
            let gpu_start = Instant::now();
            let gpu_result = run_fixture_gpu(spec, &model, gpu_mode);
            gpu_run_ms = Some(gpu_start.elapsed().as_secs_f64() * 1_000.0);

            match gpu_result {
                Ok(gpu_envelope) => {
                    run_ok = true;
                    publishable = Some(gpu_envelope.data.publishable);
                    gpu_fallback_events = gpu_envelope.data.provenance.fallback_events.clone();
                    gpu_solver_host_sync_count =
                        Some(gpu_envelope.data.provenance.solver_host_sync_count);
                    gpu_solver_device_apply_k_ratio =
                        Some(gpu_envelope.data.provenance.solver_device_apply_k_ratio);
                    gpu_solver_backend = Some(gpu_envelope.data.provenance.solver_backend.clone());
                    gpu_speedup_ratio = match (cpu_run_ms, gpu_run_ms) {
                        (Some(cpu_ms), Some(gpu_ms)) if gpu_ms > 0.0 => Some(cpu_ms / gpu_ms),
                        _ => None,
                    };
                    let cache_hits = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_hits",
                    );
                    let cache_misses = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_misses",
                    );
                    let cache_entries = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_CACHE",
                        "prepared_cache_entries",
                    );
                    gpu_transient_cache_misses = cache_misses;
                    gpu_transient_cache_entries = cache_entries;
                    gpu_transient_cache_hit_ratio = match (cache_hits, cache_misses) {
                        (Some(hits), Some(misses)) if (hits + misses) > 0.0 => {
                            Some(hits / (hits + misses))
                        }
                        _ => None,
                    };
                    gpu_solver_prepared_build_ms = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "prepared_build_ms",
                    )
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "prepared_build_ms",
                        )
                    });
                    gpu_solver_solve_ms = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "solve_ms",
                    )
                    .or_else(|| {
                        diagnostic_metric(&gpu_envelope.data, "FEA_TRANSIENT_COST", "solve_ms")
                    });
                    gpu_solver_fallback_apply_count = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_MODAL_COST",
                        "fallback_apply_count",
                    )
                    .or_else(|| {
                        diagnostic_metric(
                            &gpu_envelope.data,
                            "FEA_TRANSIENT_COST",
                            "fallback_apply_count",
                        )
                    });
                    let transient_adapt_scale_min = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_min",
                    );
                    let transient_adapt_scale_max = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_max",
                    );
                    let transient_adapt_scale_mean = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "scale_mean",
                    );
                    let transient_adapt_decrease_steps = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_ADAPTIVITY",
                        "decrease_steps",
                    );
                    let transient_bucket_rel_tolerance = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_BUCKETING",
                        "rel_tolerance",
                    );
                    let transient_physics_jump_ratio = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "max_step_l2_jump_ratio",
                    );
                    let transient_physics_nonfinite = diagnostic_metric(
                        &gpu_envelope.data,
                        "FEA_TRANSIENT_PHYSICS",
                        "nonfinite_displacement_count",
                    );

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

                    if let Some(expected_solver_backend) = spec.expected_solver_backend {
                        if gpu_envelope.data.provenance.solver_backend != expected_solver_backend {
                            failures.push(format!(
                                "solver_backend mismatch for fixture {}: expected={} got={}",
                                spec.id,
                                expected_solver_backend,
                                gpu_envelope.data.provenance.solver_backend
                            ));
                        }
                    }
                    if let Some(min_speedup_ratio) = spec.min_gpu_speedup_ratio {
                        let observed = gpu_speedup_ratio.unwrap_or(0.0);
                        if observed < min_speedup_ratio {
                            failures.push(format!(
                                "gpu speedup ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_speedup_ratio
                            ));
                        }
                    }
                    if let Some(min_cache_hit_ratio) = spec.min_transient_cache_hit_ratio {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_hit_ratio",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_hit_ratio,
                            Some(min_cache_hit_ratio),
                            None,
                        );
                    }
                    if let Some(max_cache_misses) = spec.max_transient_cache_misses {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_prepared_cache_misses",
                            "FEA_TRANSIENT_CACHE",
                            gpu_transient_cache_misses,
                            None,
                            Some(max_cache_misses),
                        );
                    }
                    if spec.id == "transient_long_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_min",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_min,
                            Some(0.65),
                            None,
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_max",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_max,
                            None,
                            Some(1.35),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_scale_mean",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_scale_mean,
                            Some(0.8),
                            Some(1.2),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_adapt_decrease_steps",
                            "FEA_TRANSIENT_ADAPTIVITY",
                            transient_adapt_decrease_steps,
                            None,
                            Some(16.0),
                        );
                        let requested_bucket_tol = std::env::var("RUNMAT_TRANSIENT_DT_BUCKET_REL_TOL")
                            .ok()
                            .and_then(|value| value.parse::<f64>().ok())
                            .unwrap_or(0.0)
                            .max(0.0);
                        if requested_bucket_tol > 0.0 {
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_rel_tolerance",
                                "FEA_TRANSIENT_BUCKETING",
                                transient_bucket_rel_tolerance,
                                Some(requested_bucket_tol * 0.99),
                                Some(requested_bucket_tol * 1.01 + 1.0e-12),
                            );
                            push_threshold_assertion(
                                spec.id,
                                &mut threshold_assertions,
                                &mut failures,
                                "transient_bucket_quality_residual",
                                "FEA_TRANSIENT_STABILITY",
                                diagnostic_metric(
                                    &gpu_envelope.data,
                                    "FEA_TRANSIENT_STABILITY",
                                    "max_residual_norm",
                                ),
                                None,
                                Some(1.0e-2),
                            );
                        }
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(3.5),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }
                    if spec.id == "transient_shock_gpu_provider" {
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_jump_ratio",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_jump_ratio,
                            None,
                            Some(4.0),
                        );
                        push_threshold_assertion(
                            spec.id,
                            &mut threshold_assertions,
                            &mut failures,
                            "transient_shock_physics_nonfinite_count",
                            "FEA_TRANSIENT_PHYSICS",
                            transient_physics_nonfinite,
                            None,
                            Some(0.0),
                        );
                    }

                    let gpu_results = analysis_results_op(
                        &gpu_envelope.data,
                        AnalysisResultsQuery {
                            include_fields: vec!["displacement".to_string()],
                            include_diagnostics: false,
                            include_modal_results: false,
                            mode_indices: Vec::new(),
                            include_transient_results: false,
                            transient_snapshot_indices: Vec::new(),
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
                                gpu_solver_host_sync_count,
                                gpu_solver_device_apply_k_ratio,
                                gpu_speedup_ratio,
                                gpu_solver_backend,
                                gpu_transient_cache_hit_ratio,
                                gpu_transient_cache_misses,
                                gpu_transient_cache_entries,
                                gpu_solver_prepared_build_ms,
                                gpu_solver_solve_ms,
                                gpu_solver_fallback_apply_count,
                                publishable,
                                parity,
                                threshold_assertions,
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
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
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

                    if let Some(max_sync) = spec.max_solver_host_sync_count {
                        let observed = gpu_envelope.data.provenance.solver_host_sync_count;
                        if observed > max_sync {
                            failures.push(format!(
                                "solver_host_sync_count exceeded for fixture {}: observed={} limit={}",
                                spec.id, observed, max_sync
                            ));
                        }
                    }
                    if let Some(min_ratio) = spec.min_solver_device_apply_k_ratio {
                        let observed = gpu_envelope.data.provenance.solver_device_apply_k_ratio;
                        if observed < min_ratio {
                            failures.push(format!(
                                "solver_device_apply_k_ratio below target for fixture {}: observed={} min={}",
                                spec.id, observed, min_ratio
                            ));
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
                                include_modal_results: false,
                                mode_indices: Vec::new(),
                                include_transient_results: false,
                                transient_snapshot_indices: Vec::new(),
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
                                    gpu_solver_host_sync_count,
                                    gpu_solver_device_apply_k_ratio,
                                    gpu_speedup_ratio,
                                    gpu_solver_backend,
                                    gpu_transient_cache_hit_ratio,
                                    gpu_transient_cache_misses,
                                    gpu_transient_cache_entries,
                                    gpu_solver_prepared_build_ms,
                                    gpu_solver_solve_ms,
                                    gpu_solver_fallback_apply_count,
                                    publishable,
                                    parity,
                                    threshold_assertions,
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
        let result = run_fixture_cpu(spec, &model);
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
        gpu_solver_host_sync_count,
        gpu_solver_device_apply_k_ratio,
        gpu_speedup_ratio,
        gpu_solver_backend,
        gpu_transient_cache_hit_ratio,
        gpu_transient_cache_misses,
        gpu_transient_cache_entries,
        gpu_solver_prepared_build_ms,
        gpu_solver_solve_ms,
        gpu_solver_fallback_apply_count,
        publishable,
        parity,
        threshold_assertions,
        failures,
    }
}

struct HarnessProvider;

fn harness_store() -> &'static Mutex<HashMap<u64, HostTensorOwned>> {
    static STORE: OnceLock<Mutex<HashMap<u64, HostTensorOwned>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_get(handle: &GpuTensorHandle) -> anyhow::Result<HostTensorOwned> {
    let guard = harness_store()
        .lock()
        .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
    guard
        .get(&handle.buffer_id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing tensor for buffer_id={}", handle.buffer_id))
}

fn store_insert(shape: Vec<usize>, data: Vec<f64>) -> anyhow::Result<GpuTensorHandle> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(anyhow::anyhow!(
            "tensor data len {} does not match shape product {}",
            data.len(),
            expected
        ));
    }
    static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(5000);
    let buffer_id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
    let mut guard = harness_store()
        .lock()
        .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
    guard.insert(buffer_id, HostTensorOwned { data, shape: shape.clone() });
    Ok(GpuTensorHandle {
        shape,
        device_id: 21,
        buffer_id,
    })
}

fn elementwise_binary(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    op: impl Fn(f64, f64) -> f64,
) -> anyhow::Result<GpuTensorHandle> {
    let ah = store_get(a)?;
    let bh = store_get(b)?;
    if ah.shape != bh.shape {
        return Err(anyhow::anyhow!("shape mismatch in elementwise operation"));
    }
    let data = ah
        .data
        .iter()
        .zip(bh.data.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    store_insert(ah.shape, data)
}

impl AccelProvider for HarnessProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        store_insert(host.shape.to_vec(), host.data.to_vec())
    }

    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move { store_get(h) })
    }

    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()> {
        let mut guard = harness_store()
            .lock()
            .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
        guard.remove(&h.buffer_id);
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

    fn elem_add<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x + y) })
    }

    fn elem_sub<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x - y) })
    }

    fn elem_mul<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { elementwise_binary(a, b, |x, y| x * y) })
    }

    fn scalar_mul(&self, a: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        let ah = store_get(a)?;
        let data = ah.data.iter().map(|&x| x * scalar).collect();
        store_insert(ah.shape, data)
    }

    fn reduce_sum<'a>(&'a self, a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let ah = store_get(a)?;
            let sum = ah.data.iter().sum::<f64>();
            store_insert(vec![1], vec![sum])
        })
    }

    fn read_scalar(&self, h: &GpuTensorHandle, linear_index: usize) -> anyhow::Result<f64> {
        let ah = store_get(h)?;
        ah.data
            .get(linear_index)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("read_scalar index out of bounds"))
    }

    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        indices: &[u32],
        output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let src = store_get(source)?;
        let out_len: usize = output_shape.iter().product();
        if indices.len() != out_len {
            return Err(anyhow::anyhow!(
                "indices len {} does not match output len {}",
                indices.len(),
                out_len
            ));
        }
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            let idx = idx as usize;
            let value = src
                .data
                .get(idx)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("gather index out of bounds"))?;
            out.push(value);
        }
        store_insert(output_shape.to_vec(), out)
    }

    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        indices: &[u32],
        values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        let values_host = store_get(values)?;
        if values_host.data.len() != indices.len() {
            return Err(anyhow::anyhow!(
                "scatter values len {} != indices len {}",
                values_host.data.len(),
                indices.len()
            ));
        }

        let mut guard = harness_store()
            .lock()
            .map_err(|_| anyhow::anyhow!("harness store lock poisoned"))?;
        let target_host = guard
            .get_mut(&target.buffer_id)
            .ok_or_else(|| anyhow::anyhow!("missing target tensor for scatter"))?;
        for (i, &idx) in indices.iter().enumerate() {
            let idx = idx as usize;
            if idx >= target_host.data.len() {
                return Err(anyhow::anyhow!("scatter index out of bounds"));
            }
            target_host.data[idx] = values_host.data[i];
        }
        Ok(())
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
    let rolling_dir = std::env::var("RUNMAT_ANALYSIS_BASELINE_DIR")
        .ok()
        .map(PathBuf::from);
    let rolling_window = std::env::var("RUNMAT_ANALYSIS_BASELINE_WINDOW")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let enforce_explicit = std::env::var("RUNMAT_ANALYSIS_ENFORCE_BASELINE")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let enforce = enforce_explicit || enforce_on_protected_branch();
    let max_slowdown_ratio = std::env::var("RUNMAT_ANALYSIS_MAX_SLOWDOWN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.5);
    let max_cost_slowdown_ratio = std::env::var("RUNMAT_ANALYSIS_MAX_COST_SLOWDOWN_RATIO")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(1.75);
    let min_speedup_retention = std::env::var("RUNMAT_ANALYSIS_MIN_SPEEDUP_RETENTION")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.75);
    BaselineConfig {
        path,
        rolling_dir,
        rolling_window,
        enforce,
        max_slowdown_ratio,
        max_cost_slowdown_ratio,
        min_speedup_retention,
    }
}

fn enforce_on_protected_branch() -> bool {
    let opt_in = std::env::var("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !opt_in {
        return false;
    }
    let ci = std::env::var("CI")
        .ok()
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !ci {
        return false;
    }
    let branch = std::env::var("GITHUB_REF_NAME")
        .or_else(|_| std::env::var("CI_COMMIT_REF_NAME"))
        .unwrap_or_default();
    if branch.is_empty() {
        return false;
    }
    let protected = std::env::var("RUNMAT_ANALYSIS_PROTECTED_BRANCHES")
        .unwrap_or_else(|_| "main,master".to_string());
    protected
        .split(',')
        .map(|value| value.trim())
        .any(|value| !value.is_empty() && value == branch)
}

fn load_baseline_report(path: &PathBuf) -> Result<BenchmarkConformanceReport, String> {
    let bytes = fs::read(path).map_err(|err| format!("failed to read baseline report: {err}"))?;
    serde_json::from_slice::<BenchmarkConformanceReport>(&bytes)
        .map_err(|err| format!("failed to parse baseline report: {err}"))
}

fn load_rolling_reports(
    dir: &PathBuf,
    limit: usize,
    exclude_path: &PathBuf,
) -> Result<Vec<BenchmarkConformanceReport>, String> {
    let entries = fs::read_dir(dir).map_err(|err| format!("failed to read baseline dir: {err}"))?;
    let mut json_paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read baseline dir entry: {err}"))?;
        let path = entry.path();
        if path == *exclude_path {
            continue;
        }
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let modified = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .map_err(|err| format!("failed to stat baseline report {}: {err}", path.display()))?;
        json_paths.push((path, modified));
    }
    json_paths.sort_by(|a, b| b.1.cmp(&a.1));

    let mut reports = Vec::new();
    for (path, _) in json_paths.into_iter().take(limit) {
        if let Ok(report) = load_baseline_report(&path) {
            reports.push(report);
        }
    }
    Ok(reports)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) * 0.5)
    }
}

fn check_rolling_baseline_drift(
    history: &[BenchmarkConformanceReport],
    current: &BenchmarkConformanceReport,
    max_slowdown_ratio: f64,
    max_cost_slowdown_ratio: f64,
    min_speedup_retention: f64,
    failures: &mut Vec<String>,
) {
    for fixture_id in ROLLING_TARGET_FIXTURES {
        let Some(current_record) = current.records.iter().find(|record| record.fixture_id == *fixture_id)
        else {
            continue;
        };

        let mut gpu_ms_history = Vec::new();
        let mut speedup_history = Vec::new();
        for report in history {
            if let Some(record) = report.records.iter().find(|value| value.fixture_id == *fixture_id) {
                if let Some(gpu_ms) = record.gpu_run_ms {
                    if gpu_ms > 0.0 {
                        gpu_ms_history.push(gpu_ms);
                    }
                }
                if let Some(speedup) = record.gpu_speedup_ratio {
                    if speedup > 0.0 {
                        speedup_history.push(speedup);
                    }
                }
            }
        }

        if gpu_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (current_record.gpu_run_ms, median(&mut gpu_ms_history)) {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline slowdown exceeded for fixture={} backend=gpu ratio={ratio:.3} limit={max_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }

        if speedup_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_speedup_ratio,
                median(&mut speedup_history),
            ) {
                let min_allowed = base_median * min_speedup_retention;
                if now < min_allowed {
                    failures.push(format!(
                        "rolling baseline speedup retention failed for fixture={} speedup={now:.3} min_allowed={min_allowed:.3} (median_speedup={base_median:.3}, retention={min_speedup_retention:.3})",
                        fixture_id
                    ));
                }
            }
        }

        let mut solve_ms_history = Vec::new();
        let mut prepared_build_ms_history = Vec::new();
        for report in history {
            if let Some(record) = report.records.iter().find(|value| value.fixture_id == *fixture_id) {
                if let Some(value) = record.gpu_solver_solve_ms {
                    if value > 0.0 {
                        solve_ms_history.push(value);
                    }
                }
                if let Some(value) = record.gpu_solver_prepared_build_ms {
                    if value > 0.0 {
                        prepared_build_ms_history.push(value);
                    }
                }
            }
        }
        if solve_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_solver_solve_ms,
                median(&mut solve_ms_history),
            ) {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_cost_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline solve_ms slowdown exceeded for fixture={} ratio={ratio:.3} limit={max_cost_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }
        if prepared_build_ms_history.len() >= 2 {
            if let (Some(now), Some(base_median)) = (
                current_record.gpu_solver_prepared_build_ms,
                median(&mut prepared_build_ms_history),
            ) {
                let ratio = now / base_median.max(1.0e-12);
                if ratio > max_cost_slowdown_ratio {
                    failures.push(format!(
                        "rolling baseline prepared_build_ms slowdown exceeded for fixture={} ratio={ratio:.3} limit={max_cost_slowdown_ratio:.3} (median_ms={base_median:.3}, current_ms={now:.3})",
                        fixture_id
                    ));
                }
            }
        }
    }
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
