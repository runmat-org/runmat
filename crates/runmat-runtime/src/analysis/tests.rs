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
    GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, SourceGeometry, SourceGeometryKind,
    TessellationProfile, UnitSystem,
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
fn analysis_create_model_maps_unsupported_profile_error() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    for profile in [
        AnalysisCreateModelProfile::TransientStructural,
        AnalysisCreateModelProfile::NonlinearStructural,
    ] {
        let err = analysis_create_model_op(
            &geometry,
            AnalysisCreateModelIntentSpec {
                model_id: "unsupported_model".to_string(),
                profile,
            },
            OperationContext::new(None, None),
        )
        .expect_err("profile should be unsupported");

        assert_eq!(err.error_code, "ANALYSIS_CREATE_MODEL_PROFILE_UNSUPPORTED");
        assert_eq!(err.operation, "analysis.create_model");
        assert_eq!(err.op_version, "analysis.create_model/v1");
    }
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
fn analysis_run_modal_returns_degraded_placeholder_result() {
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
    .expect("modal run should produce placeholder result");

    assert_eq!(envelope.operation, "analysis.run_modal");
    assert_eq!(envelope.op_version, "analysis.run_modal/v1");
    assert_eq!(envelope.data.run.solver_method, "modal_placeholder_from_linear_static");
    assert_eq!(envelope.data.provenance.solver_method, "modal_placeholder_from_linear_static");
    assert_eq!(envelope.data.run_status, RunStatus::Degraded);
    assert!(!envelope.data.publishable);
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal placeholder payload should exist");
    assert_eq!(modal.eigenvalues_hz.len(), 1);
    assert_eq!(modal.mode_shapes.len(), 1);
    assert_eq!(modal.mode_shapes[0].field_id, "mode_shape_1");
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalPlaceholder));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_PLACEHOLDER"));
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
    .expect("modal run placeholder should succeed");

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
    assert_eq!(modal.eigenvalues_hz.len(), 1);
    assert_eq!(modal.mode_shapes.len(), 1);
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
    .expect("modal run placeholder should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: false,
            mode_indices: Vec::new(),
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
    .expect("modal run placeholder should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_diagnostics: true,
            include_modal_results: true,
            mode_indices: vec![10],
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown mode index");

    assert_eq!(err.error_code, "ANALYSIS_RESULTS_MODE_NOT_FOUND");
    assert_eq!(err.operation, "analysis.results");
    assert_eq!(err.op_version, "analysis.results/v1");
}
