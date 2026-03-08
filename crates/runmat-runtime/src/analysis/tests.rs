use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, ApiDeviceInfo, GpuTensorHandle, HostTensorOwned,
    HostTensorView,
};
use runmat_analysis_core::{
    AnalysisFieldValues, AnalysisModel, AnalysisModelId, AnalysisStep, AnalysisStepKind,
    BoundaryCondition, BoundaryConditionKind, LoadCase, LoadKind, MaterialModel, ReferenceFrame,
};
use runmat_analysis_fea::ComputeBackend;
use runmat_geometry_core::{
    GeometryAsset, GeometrySource, MaterialEvidence, MaterialEvidenceConfidence, MeshDescriptor,
    MeshKind, Region, SourceGeometry, SourceGeometryKind, TessellationProfile, UnitSystem,
};

use super::*;

fn analysis_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sample_model() -> AnalysisModel {
    AnalysisModel {
        model_id: AnalysisModelId("beam_model".to_string()),
        geometry_id: "geo:beam".to_string(),
        geometry_revision: 1,
        units: UnitSystem::Meter,
        frame: ReferenceFrame::Global,
        materials: vec![MaterialModel {
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            youngs_modulus_pa: 200e9,
            poisson_ratio: 0.3,
        }],
        material_assignments: Vec::new(),
        boundary_conditions: vec![BoundaryCondition {
            bc_id: "bc_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }],
        loads: vec![LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -1000.0,
                fz: 0.0,
            },
        }],
        steps: vec![AnalysisStep {
            step_id: "step_static".to_string(),
            kind: AnalysisStepKind::Static,
        }],
    }
}

fn sample_geometry_asset() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo:beam".to_string(),
        source: GeometrySource {
            path: "/fixtures/beam.stl".to_string(),
            sha256: "hash-beam".to_string(),
            importer_version: "stl/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 2,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 3,
            element_count: 1,
        }],
        regions: Vec::new(),
        diagnostics: Vec::new(),
    }
}

fn sample_step_like_geometry_asset() -> GeometryAsset {
    let mut asset = sample_geometry_asset();
    asset.source_geometry.kind = SourceGeometryKind::Cad;
    asset.source_geometry.material_evidence = vec![MaterialEvidence {
        source_key: "STEP:MATERIAL".to_string(),
        normalized_key: "material_name".to_string(),
        value: "Aluminum 6061".to_string(),
        confidence: MaterialEvidenceConfidence::High,
        unit_basis: None,
        assumptions: vec!["imported".to_string()],
    }];
    asset.regions = vec![
        Region {
            region_id: "region_root".to_string(),
            name: "Base_Mount".to_string(),
            tag: Some("fixed".to_string()),
        },
        Region {
            region_id: "region_tip".to_string(),
            name: "Tip_Load".to_string(),
            tag: Some("load".to_string()),
        },
    ];
    asset
}

#[test]
fn analysis_create_model_returns_v1_envelope() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_geo".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
        },
        OperationContext::new(Some("trace-create-1".to_string()), None),
    )
    .expect("create model should pass");

    assert_eq!(envelope.operation, "analysis.create_model");
    assert_eq!(envelope.op_version, "analysis.create_model/v1");
    assert_eq!(envelope.data.model_id.0, "model_from_geo");
    assert_eq!(envelope.data.geometry_id, "geo:beam");
    assert_eq!(envelope.data.geometry_revision, 2);
    assert_eq!(envelope.data.units, UnitSystem::Meter);
    assert_eq!(envelope.data.frame, ReferenceFrame::Global);
    assert!(!envelope.data.materials.is_empty());
    assert!(!envelope.data.boundary_conditions.is_empty());
    assert!(!envelope.data.loads.is_empty());
    assert!(!envelope.data.steps.is_empty());
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Static);
}

#[test]
fn transient_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisTransientRunOptions::coarse();
    let balanced = AnalysisTransientRunOptions::balanced();
    let production = AnalysisTransientRunOptions::production_recommended();
    let high_accuracy = AnalysisTransientRunOptions::high_accuracy();

    assert!(coarse.step_count < balanced.step_count);
    assert!(balanced.step_count < high_accuracy.step_count);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance > high_accuracy.tolerance);

    assert!(coarse.time_step_s > balanced.time_step_s);
    assert!(balanced.time_step_s > high_accuracy.time_step_s);

    assert_eq!(production.quality_policy, QualityPolicy::Balanced);
    assert!(production.deterministic_mode);
    assert_eq!(production.precision_mode, PrecisionMode::Fp64);
    assert_eq!(production.dt_bucket_rel_tolerance, 0.01);
}

#[test]
fn modal_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisModalRunOptions::coarse();
    let balanced = AnalysisModalRunOptions::balanced();
    let high_accuracy = AnalysisModalRunOptions::high_accuracy();

    assert!(coarse.mode_count < balanced.mode_count);
    assert!(balanced.mode_count < high_accuracy.mode_count);
    assert!(coarse.residual_warn_threshold > balanced.residual_warn_threshold);
    assert!(balanced.residual_warn_threshold > high_accuracy.residual_warn_threshold);
}

#[test]
fn analysis_create_model_maps_invalid_intent_error() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let err = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "   ".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
        },
        OperationContext::new(None, None),
    )
    .expect_err("create model should fail");

    assert_eq!(err.error_code, "ANALYSIS_CREATE_MODEL_INVALID_INTENT");
    assert_eq!(err.operation, "analysis.create_model");
    assert_eq!(err.op_version, "analysis.create_model/v1");
}

#[test]
fn analysis_create_model_supports_nonlinear_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "nonlinear_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear profile should be supported");

    assert_eq!(envelope.data.model_id.0, "nonlinear_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Nonlinear);
    assert_eq!(envelope.data.loads[0].load_id, "load_default_nonlinear_force");
}

#[test]
fn analysis_create_model_supports_transient_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "transient_model".to_string(),
            profile: AnalysisCreateModelProfile::TransientStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("transient profile should be supported");

    assert_eq!(envelope.data.model_id.0, "transient_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Transient);
    assert_eq!(envelope.data.loads[0].load_id, "load_default_transient_force");
}

#[test]
fn analysis_create_model_supports_modal_profile_template() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal profile should be supported");

    assert_eq!(envelope.data.model_id.0, "modal_model");
    assert_eq!(envelope.data.steps[0].kind, AnalysisStepKind::Modal);
    assert_eq!(envelope.data.loads[0].load_id, "load_default_modal_seed");
}

#[test]
fn analysis_create_model_infers_materials_and_assignments_from_geometry_evidence() {
    let _guard = analysis_test_guard();
    let geometry = sample_step_like_geometry_asset();
    let envelope = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "model_from_step_like".to_string(),
            profile: AnalysisCreateModelProfile::LinearStaticStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("create model should succeed");

    assert!(envelope
        .data
        .materials
        .iter()
        .any(|material| material.material_id == "mat_aluminum"));
    assert_eq!(envelope.data.material_assignments.len(), 2);
    assert!(envelope
        .data
        .material_assignments
        .iter()
        .all(|assignment| assignment.assigned_material_id == "mat_aluminum"));
    assert_eq!(envelope.data.boundary_conditions[0].region_id, "region_root");
    assert_eq!(envelope.data.loads[0].region_id, "region_tip");
}

#[test]
fn analysis_validate_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context = OperationContext::new(Some("trace-a1".to_string()), Some("request-a1".to_string()));
    let envelope = analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
        .expect("validation should pass");

    assert_eq!(envelope.operation, "analysis.validate");
    assert_eq!(envelope.op_version, "analysis.validate/v1");
    assert!(envelope.data.valid);
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-a1"));
}

#[test]
fn analysis_validate_maps_typed_error_code() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.materials.clear();
    let context = OperationContext::new(None, None);
    let error =
        analysis_validate(&model, UnitSystem::Meter, &ReferenceFrame::Global, context)
            .expect_err("validation should fail");

    assert_eq!(error.error_code, "ANALYSIS_VALIDATION_MISSING_MATERIALS");
    assert_eq!(error.operation, "analysis.validate");
    assert_eq!(error.op_version, "analysis.validate/v1");
}

#[test]
fn analysis_run_linear_static_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context = OperationContext::new(Some("trace-a2".to_string()), Some("request-a2".to_string()));
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
        },
        context,
    )
    .expect("run should pass");

    assert_eq!(envelope.operation, "analysis.run_linear_static");
    assert_eq!(envelope.op_version, "analysis.run_linear_static/v1");
    assert_eq!(envelope.data.run.backend, ComputeBackend::Cpu);
    assert!(!envelope.data.run.displacement_field.is_empty());
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(envelope.data.modal_results.is_none());
    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert!(envelope.data.provenance.deterministic_mode);
    assert_eq!(envelope.data.provenance.precision_mode, "fp64");
    assert_eq!(envelope.data.provenance.solver_method, "matrix_free_pcg");
    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
    assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(envelope.data.provenance.solver_host_sync_count, 0);
}

#[test]
fn gpu_run_without_provider_records_fallback_event() {
    let _guard = analysis_test_guard();
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let model = sample_model();
    let envelope =
        analysis_run_linear_static_op(&model, ComputeBackend::Gpu, OperationContext::new(None, None))
            .expect("run should pass");

    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
        assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
        assert!(matches!(
            envelope.data.run.displacement_field.values,
            AnalysisFieldValues::HostF64(_)
        ));
    } else {
        assert_eq!(envelope.data.provenance.solver_backend, "runtime_tensor");
    }
}

#[test]
fn gpu_run_with_provider_emits_device_refs() {
    let _guard = analysis_test_guard();
    static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1000);

    struct AnalysisTestProvider;

    impl AccelProvider for AnalysisTestProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: 7,
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
            "analysis-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            7
        }

        fn device_info_struct(&self) -> ApiDeviceInfo {
            ApiDeviceInfo {
                device_id: 7,
                name: "analysis-test-provider".to_string(),
                vendor: "runmat-tests".to_string(),
                memory_bytes: None,
                backend: Some("test_gpu".to_string()),
            }
        }
    }

    static PROVIDER: AnalysisTestProvider = AnalysisTestProvider;
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope =
        analysis_run_linear_static_op(&model, ComputeBackend::Gpu, OperationContext::new(None, None))
            .expect("run should pass");

    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("BACKEND_NO_PROVIDER")
            || event.starts_with("BACKEND_UPLOAD_FAILED")));
    assert!(matches!(
        envelope.data.provenance.solver_backend.as_str(),
        "runtime_tensor" | "cpu_reference"
    ));
    if envelope.data.provenance.solver_backend == "cpu_reference" {
        assert!(envelope
            .data
            .provenance
            .fallback_events
            .iter()
            .any(|event| event.starts_with("SOLVER_BACKEND_FALLBACK")));
    }
    assert!(
        (0.0..=1.0).contains(&envelope.data.provenance.solver_device_apply_k_ratio),
        "ratio must be in [0,1]"
    );
    assert!(matches!(
        envelope.data.run.displacement_field.values,
        AnalysisFieldValues::DeviceRef(_)
    ));
    assert!(matches!(
        envelope.data.run.von_mises_field.values,
        AnalysisFieldValues::DeviceRef(_)
    ));
}

#[test]
fn analysis_results_returns_filtered_fields_and_metadata() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["displacement".to_string()],
            include_diagnostics: false,
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-2".to_string()), None),
    )
    .expect("results should pass");

    assert_eq!(results.operation, "analysis.results");
    assert_eq!(results.op_version, "analysis.results/v1");
    assert_eq!(results.data.fields.len(), 1);
    assert_eq!(results.data.fields[0].field_id, "displacement");
    assert!(results.data.diagnostics.is_none());
    assert_eq!(results.data.summary.field_count, 1);
    assert_eq!(results.data.summary.mode_count, 0);
    assert!(results.data.summary.available_mode_indices.is_empty());
    assert_eq!(results.data.summary.min_frequency_hz, None);
    assert_eq!(results.data.summary.max_frequency_hz, None);
    assert_eq!(results.data.summary.max_modal_residual_norm, None);
    assert_eq!(results.data.summary.first_mode_converged, None);
    assert_eq!(results.data.summary.snapshot_count, 0);
    assert_eq!(results.data.summary.time_start_s, None);
    assert_eq!(results.data.summary.time_end_s, None);
    assert_eq!(results.data.summary.max_transient_residual_norm, None);
    assert_eq!(results.data.summary.final_step_converged, None);
}

#[test]
fn analysis_results_unknown_field_maps_typed_error() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-3".to_string()), None),
    )
    .expect("run should pass");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec!["strain_energy".to_string()],
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-4".to_string()), None),
    )
    .expect_err("results should fail");

    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
    assert_eq!(err.error_code, "ANALYSIS_RESULTS_FIELD_NOT_FOUND");
}

#[test]
fn analysis_results_by_run_id_roundtrip_works() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let model = sample_model();
    let run = analysis_run_linear_static_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(Some("trace-results-by-id-run".to_string()), None),
    )
    .expect("run should pass");

    let fetched = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-fetch".to_string()), None),
    )
    .expect("results by id should pass");

    assert_eq!(fetched.operation, "analysis.results");
    assert_eq!(fetched.op_version, "analysis.results/v1");
    assert_eq!(fetched.data.summary.field_count, 2);
    assert_eq!(fetched.data.summary.mode_count, 0);
    assert!(fetched.data.summary.available_mode_indices.is_empty());
    assert_eq!(fetched.data.summary.min_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_modal_residual_norm, None);
    assert_eq!(fetched.data.summary.first_mode_converged, None);
    assert_eq!(fetched.data.summary.snapshot_count, 0);
    assert_eq!(fetched.data.summary.time_start_s, None);
    assert_eq!(fetched.data.summary.time_end_s, None);
    assert_eq!(fetched.data.summary.max_transient_residual_norm, None);
    assert_eq!(fetched.data.summary.final_step_converged, None);

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let err = analysis_results_by_run_id_op(
        "run_missing",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-missing".to_string()), None),
    )
    .expect_err("missing run id should fail");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_RUN_NOT_FOUND");
    storage::reset_artifact_store_for_tests();
}

#[test]
fn requested_preconditioner_fallback_is_recorded() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Amg,
            quality_policy: QualityPolicy::Balanced,
        },
        OperationContext::new(Some("trace-preconditioner-fallback".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "jacobi");
    assert!(envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn ilu_preconditioner_request_is_honored_without_fallback() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Ilu,
            quality_policy: QualityPolicy::Balanced,
        },
        OperationContext::new(Some("trace-preconditioner-ilu".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.provenance.preconditioner, "ilu0");
    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.starts_with("SOLVER_PRECONDITIONER_FALLBACK")));
}

#[test]
fn quality_policy_exploratory_allows_publishable_warn_path() {
    let _guard = analysis_test_guard();
    let model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::MultiMaterialAssembly,
    );
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Exploratory,
        },
        OperationContext::new(Some("trace-quality-policy-exploratory".to_string()), None),
    )
    .expect("run should succeed");

    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    assert_eq!(envelope.data.provenance.quality_policy, "exploratory");
}

#[test]
fn quality_policy_balanced_allows_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
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
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
        },
        OperationContext::new(Some("trace-quality-policy-balanced".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
}

#[test]
fn quality_policy_strict_rejects_publishable_with_quality_reasons() {
    let _guard = analysis_test_guard();

    struct UploadFailProvider;

    impl AccelProvider for UploadFailProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("forced-upload-failure"))
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
            "upload-fail-provider".to_string()
        }
    }

    static PROVIDER: UploadFailProvider = UploadFailProvider;
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(&PROVIDER));

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Gpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Strict,
        },
        OperationContext::new(Some("trace-quality-policy-strict".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(!envelope.data.publishable);
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::FieldPromotionFallback));
    assert_eq!(envelope.data.provenance.quality_policy, "strict");
}

#[test]
fn analysis_run_modal_rejects_models_without_modal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_modal_op(&model, ComputeBackend::Cpu, OperationContext::new(None, None))
        .expect_err("modal run should fail for missing modal step");

    assert_eq!(err.operation, "analysis.run_modal");
    assert_eq!(err.op_version, "analysis.run_modal/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_MODAL_INVALID_MODEL");
}

#[test]
fn analysis_run_transient_rejects_models_without_transient_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("transient run should fail for missing transient step");

    assert_eq!(err.operation, "analysis.run_transient");
    assert_eq!(err.op_version, "analysis.run_transient/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_TRANSIENT_INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_rejects_models_without_nonlinear_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject models without nonlinear step");
    assert_eq!(err.operation, "analysis.run_nonlinear");
    assert_eq!(err.op_version, "analysis.run_nonlinear/v1");
    assert_eq!(err.error_code, "ANALYSIS_RUN_NONLINEAR_INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_returns_native_nonlinear_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let envelope = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            increment_count: 16,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    assert_eq!(envelope.operation, "analysis.run_nonlinear");
    assert_eq!(envelope.op_version, "analysis.run_nonlinear/v1");
    let nonlinear = envelope
        .data
        .nonlinear_results
        .as_ref()
        .expect("nonlinear payload should exist");
    assert_eq!(nonlinear.method, NonlinearMethod::IncrementalNewtonRaphson);
    assert_eq!(nonlinear.load_factors.len(), 16);
    assert_eq!(nonlinear.load_factors.len(), nonlinear.residual_norms.len());
    assert_eq!(nonlinear.residual_norms.len(), nonlinear.iteration_counts.len());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
}

#[test]
fn analysis_results_query_can_exclude_nonlinear_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: AnalysisStepKind::Nonlinear,
    }];
    let run = analysis_run_nonlinear_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: false,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.nonlinear_results.is_none());
    assert!(results.data.summary.increment_count > 0);
    assert!(results.data.summary.max_nonlinear_residual_norm.is_some());
    assert!(results.data.summary.final_increment_converged.is_some());
}

#[test]
fn analysis_run_transient_returns_native_transient_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert_eq!(envelope.operation, "analysis.run_transient");
    assert_eq!(envelope.op_version, "analysis.run_transient/v1");
    assert_eq!(envelope.data.run.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.provenance.solver_method, "implicit_euler_pcg");
    assert_eq!(envelope.data.run_status, RunStatus::Publishable);
    assert!(envelope.data.publishable);
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::TransientPlaceholder));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.integration_method, TransientIntegrationMethod::ImplicitEuler);
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
}

#[test]
fn analysis_run_transient_with_options_controls_timeline() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            time_step_s: 2.0e-3,
            min_time_step_s: 2.0e-3,
            max_time_step_s: 2.0e-3,
            step_count: 3,
            max_linear_iters: 64,
            tolerance: 1.0e-8,
            residual_target: 1.0e-6,
            adaptive_time_step: false,
            max_step_retries: 0,
            adapt_min_scale: 0.8,
            adapt_max_scale: 1.25,
            adapt_growth_exponent: 0.35,
            adapt_retry_growth_cap: 1.05,
            adapt_nonconverged_shrink: 0.75,
            dt_bucket_rel_tolerance: 0.0,
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed with options");

    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.time_points_s.len(), 4);
    assert_eq!(transient.time_points_s[0], 0.0);
    assert!((transient.time_points_s[3] - 6.0e-3).abs() < 1.0e-12);
    assert_eq!(envelope.data.provenance.deterministic_mode, true);
}

#[test]
fn analysis_run_modal_returns_native_modal_result() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should produce modal result");

    assert_eq!(envelope.operation, "analysis.run_modal");
    assert_eq!(envelope.op_version, "analysis.run_modal/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        "matrix_free_subspace_iteration"
    );
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(modal.mode_shapes[0].field_id, "mode_shape_1");
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert!(modal.residual_norms.iter().all(|value| value.is_finite()));
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(
        modal.frequency_basis,
        ModalFrequencyBasis::NativeEigenSolve
    );
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalPlaceholder));
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn analysis_run_modal_with_options_controls_requested_mode_count() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_run_opts".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            mode_count: 2,
            residual_warn_threshold: 1.0e-2,
        },
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed with options");

    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(modal.eigenvalues_hz.len() > 0);
    assert!(modal.eigenvalues_hz.len() <= 2);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_results_include_modal_payload_for_modal_runs() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let modal = results
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should propagate to results");
    assert!(!modal.eigenvalues_hz.is_empty());
    assert_eq!(modal.eigenvalues_hz.len(), modal.mode_shapes.len());
    assert_eq!(modal.eigenvalues_hz.len(), modal.residual_norms.len());
    assert_eq!(modal.modal_payload_version, "modal_results/v1");
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(
        modal.frequency_basis,
        ModalFrequencyBasis::NativeEigenSolve
    );
    assert!(results.data.summary.mode_count > 0);
    assert_eq!(
        results.data.summary.mode_count,
        results.data.summary.available_mode_indices.len()
    );
    assert!(results.data.summary.min_frequency_hz.is_some());
    assert!(results.data.summary.max_frequency_hz.is_some());
    assert!(results.data.summary.max_modal_residual_norm.is_some());
    assert!(results.data.summary.first_mode_converged.is_some());
}

#[test]
fn analysis_results_query_can_exclude_modal_payload() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_filter".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: false,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.modal_results.is_none());
}

#[test]
fn analysis_results_query_rejects_unknown_modal_mode_index() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = analysis_create_model_op(
        &geometry,
        AnalysisCreateModelIntentSpec {
            model_id: "modal_model_results_index".to_string(),
            profile: AnalysisCreateModelProfile::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: vec![10],
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown mode index");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_MODE_NOT_FOUND");
    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
}

#[test]
fn analysis_results_include_transient_payload_for_transient_runs() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let transient = results
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should propagate");
    assert_eq!(transient.integration_method, TransientIntegrationMethod::ImplicitEuler);
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
}

#[test]
fn analysis_results_query_can_exclude_transient_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: false,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.transient_results.is_none());
    assert!(results.data.summary.snapshot_count > 0);
    assert_eq!(results.data.summary.time_start_s, Some(0.0));
    assert!(results.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(results.data.summary.max_transient_residual_norm.is_some());
    assert!(results.data.summary.final_step_converged.is_some());
}

#[test]
fn analysis_results_query_rejects_unknown_transient_snapshot_index() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.steps = vec![AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: AnalysisStepKind::Transient,
    }];
    let run = analysis_run_transient_op(
        &model,
        ComputeBackend::Cpu,
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: vec![999],
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown transient snapshot index");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_TRANSIENT_SNAPSHOT_NOT_FOUND");
    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
}
